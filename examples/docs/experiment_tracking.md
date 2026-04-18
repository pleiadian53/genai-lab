# Experiment Tracking with Weights & Biases

Track metrics, hyperparameters, and artifacts across runs so you can compare
experiments visually, reproduce results, and share findings. genai-lab uses
[**Weights & Biases (W&B)**](https://wandb.ai) as the default tracker.

For pure "how do I launch a long job" mechanics, see
[`running_experiments.md`](running_experiments.md). This document assumes the
nohup / run-dir discipline from there and layers tracking on top.

---

## Why W&B

- **Free for individuals** — 100 GB storage, unlimited personal projects, no
  credit card required
- **Best UX for comparison** — parallel coordinate plots, loss curve overlays,
  sample image galleries, system metrics (GPU util, memory)
- **De facto standard** in generative-AI research — every baseline you'll
  compare against (scGen, scPPDM, CPA) uses it

MLflow is also viable if you need self-hosting. genai-lab is not set up for
MLflow today; pick W&B unless you have a specific reason not to.

---

## One-Time Setup

### 1. Account + API key

1. Sign up at [wandb.ai](https://wandb.ai) (free tier)
2. Get your API key: [wandb.ai/authorize](https://wandb.ai/authorize)
3. Identify your **entity** — this is your W&B team or username. For
   genai-lab we use the team name `genomic-ai`; use your own if working
   solo.

### 2. Local config (`.env`)

Copy [`.env.example`](../../.env.example) to `.env` and fill in:

```bash
cp .env.example .env
# edit .env with your actual values
```

Required lines:

```
WANDB_API_KEY=<your-api-key>
WANDB_ENTITY=genomic-ai    # or your username
WANDB_PROJECT=genai-lab
```

`.env` is gitignored — safe to keep secrets here. The file is loaded
automatically by scripts that import `dotenv.load_dotenv()`.

### 3. Install

W&B is in the `experiment` extras group:

```bash
pip install -e ".[experiment]"      # or just: pip install wandb python-dotenv
```

### 4. Verify

```bash
python -c "import wandb; wandb.login()"
# Should print: "wandb: Currently logged in as: <you> (<entity>)"
```

### 5. Remote (pod) setup

`ops/provision_cluster.py` takes care of the pod side automatically:

- Installs `wandb` and `python-dotenv` in the pod's setup phase
- Forwards `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` from your local
  `.env` to the pod's environment

No extra action needed on the pod — when your script calls `wandb.init()`
it's already authenticated.

---

## Running a Tracked Experiment

The flow-matching baseline is wired for W&B:

```bash
# Local
python examples/flow_matching/01_mnist_flow_matching.py --wandb \
    --epochs 50 --wandb-tags baseline,mnist

# Pod (with nohup, see running_experiments.md)
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=runs/flow_matching/mnist_fm_${TS}
mkdir -p "$RUN_DIR"
nohup python examples/flow_matching/01_mnist_flow_matching.py --wandb \
    --epochs 300 --batch-size 256 --base-channels 128 --n-steps 100 \
    --wandb-run-name "mnist-fm-bc128-bs256" \
    --wandb-tags baseline,mnist \
    --checkpoint-dir "$RUN_DIR" \
    > "$RUN_DIR/training.log" 2>&1 &
echo $! > "$RUN_DIR/pid"
```

The script logs the W&B run URL to the training log; `tail -f
$RUN_DIR/training.log` to find it.

---

## What Gets Logged

The flow-matching integration logs (and any new integration *should* log):

| Signal | Where | How |
|--------|-------|-----|
| Hyperparameters | Run config | `wandb.init(config=vars(args))` |
| Per-epoch train loss | Metrics | `wandb.log({"train/loss": ...}, step=epoch)` |
| Per-epoch val loss | Metrics | `wandb.log({"val/loss": ...}, step=epoch)` |
| Learning rate | Metrics | `wandb.log({"lr": ...}, step=epoch)` |
| Final sample grid | Media | `wandb.log({"samples_final": wandb.Image(path)})` |
| Final checkpoint | Artifact | `wandb.Artifact(type="model").add_file(...)` |
| System metrics | Auto | W&B captures GPU/CPU/RAM automatically |

**Metric naming convention**: use `<split>/<metric>` (e.g., `train/loss`,
`val/accuracy`). W&B groups charts by prefix, so consistent naming gives
you side-by-side `train/*` and `val/*` panels for free.

---

## Integrating a New Training Script

Use [`examples/flow_matching/01_mnist_flow_matching.py`](../flow_matching/01_mnist_flow_matching.py)
as the reference pattern. The steps:

### 1. Load `.env` at the top

```python
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass
```

### 2. Add CLI flags

```python
p.add_argument("--wandb", action="store_true")
p.add_argument("--wandb-run-name", type=str, default=None)
p.add_argument("--wandb-tags", type=str, default="")
```

### 3. Initialize

```python
wandb_run = None
if args.wandb:
    import wandb
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "genai-lab"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=args.wandb_run_name,
        tags=tags or None,
        config=vars(args),
        dir=str(out_dir),
    )
```

### 4. Log per-epoch via trainer callback

The `FlowMatchingTrainer.fit()` accepts an `on_epoch_end` callback. Adopt the
same pattern for other trainers: a `Callable[[dict], None]` that the trainer
invokes after each epoch with `{"epoch", "train_loss", "val_loss", "lr"}`.

```python
def _log_epoch_to_wandb(metrics):
    if wandb_run is None:
        return
    import wandb
    payload = {"train/loss": metrics["train_loss"], "lr": metrics["lr"]}
    if metrics["val_loss"] is not None:
        payload["val/loss"] = metrics["val_loss"]
    wandb.log(payload, step=metrics["epoch"])

trainer.fit(..., on_epoch_end=_log_epoch_to_wandb if wandb_run else None)
```

If a trainer doesn't have a callback hook yet, **add one**. The library
should not import `wandb` directly — tracking stays in the example script
so library code remains framework-agnostic.

### 5. Finalize + artifacts

```python
if wandb_run is not None:
    import wandb
    wandb.log({"samples_final": wandb.Image(str(grid_path))})
    wandb.run.summary["final/train_loss"] = history["train_loss"][-1]
    artifact = wandb.Artifact(
        name=f"{wandb_run.name}-model", type="model", metadata=vars(args)
    )
    artifact.add_file(str(out_dir / "model_final.pt"))
    wandb_run.log_artifact(artifact)
    wandb.finish()
```

---

## Project / Run Organization

### Project naming

Use a single project (`genai-lab`) across all topics. Separate topics via
**tags**, not projects — this lets you compare across topic boundaries when
useful (e.g., baseline CVAE vs diffusion for the same dataset).

### Run naming

`<method>-<task>-<key-hyperparam>-<key-hyperparam>`:

- `flowmatch-mnist-bc128-bs256`
- `cvae-nb-pbmc3k-lr1e3-beta0.5`
- `jepa-norman-vicreg-emb64`

Skip timestamps — W&B tracks creation time automatically.

### Tags

Tag every run. Good tag categories:

- **Status**: `baseline`, `sweep`, `debug`, `final`
- **Topic**: `mnist`, `pbmc`, `norman`, `diffusion`, `flow-matching`
- **Environment**: `local`, `pod-a40`, `pod-a100`

---

## Hyperparameter Sweeps

W&B Sweeps can orchestrate hyperparameter searches (grid, random, or
Bayesian). Minimal example:

```yaml
# sweep_flowmatch.yaml
program: examples/flow_matching/01_mnist_flow_matching.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  lr:           { distribution: log_uniform_values, min: 1e-5, max: 1e-3 }
  batch_size:   { values: [128, 256, 512] }
  base_channels:{ values: [64, 96, 128] }
command:
  - ${env}
  - python
  - ${program}
  - --wandb
  - --epochs=50
  - ${args}
```

```bash
wandb sweep sweep_flowmatch.yaml
# Copy the <sweep-id> it prints
wandb agent <sweep-id>          # run one agent per worker/GPU
```

For now, most of our work uses manual runs + tags. Reach for sweeps when
hyperparameter space is larger than ~10 runs.

---

## Offline Mode

If the pod can't reach the W&B servers (rare — RunPod allows egress by
default), run in offline mode and sync later:

```bash
# On the pod
export WANDB_MODE=offline
python your_script.py --wandb ...

# Files end up in wandb/offline-run-<timestamp>/
# Later, from a machine with network:
wandb sync wandb/offline-run-*
```

---

## Retrieval & Reproducibility

When pulling results back from the pod, `rsync` captures the local
`training.log` and checkpoints (see `running_experiments.md`). W&B keeps a
parallel record in the cloud — the two serve different purposes:

- **Local artifacts**: primary record, what you rerun/inspect offline
- **W&B run**: analytics, comparison, sharing

If a run's checkpoint is lost locally but logged as a W&B artifact, pull it
back with:

```bash
wandb artifact get <entity>/<project>/<artifact-name>:<version>
```

---

## Security Reminder

`.env` contains your API key — **never commit it**. The file is gitignored
and stays that way. If you think a key may have leaked (e.g., pasted in
chat, screenshared, committed briefly), rotate it at
[wandb.ai/authorize](https://wandb.ai/authorize) — generating a new key
invalidates the old one immediately.

---

## Related

- [`running_experiments.md`](running_experiments.md) — nohup pattern, run
  directory convention
- [`ops/README.md`](../../ops/README.md) — pod provisioning (forwards the
  W&B env vars automatically)
- [W&B Python API docs](https://docs.wandb.ai/ref/python)
- [W&B Sweeps docs](https://docs.wandb.ai/guides/sweeps)
