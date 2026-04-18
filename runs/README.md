# Runs Directory

Per-invocation training artifacts. One directory per training run.

## Not in Git

Only this `README.md` and the local `.gitignore` are tracked. All training
artifacts (logs, checkpoints, sample grids) are excluded.

## Layout

Organize as `runs/<topic>/<run-name>_<YYYYMMDD_HHMMSS>/`:

```
runs/
├── flow_matching/
│   └── mnist_fm_20260417_175639/
│       ├── training.log         # nohup stdout + stderr
│       ├── pid                  # PID file (for kill/status after SSH reconnect)
│       ├── checkpoint_*.pt      # periodic checkpoints (optimizer + weights)
│       ├── model_final.pt       # final weights only (smaller, inference-ready)
│       ├── samples_final.png    # end-of-training sample grid
│       └── config.json          # optional: snapshot of CLI args
├── perturbation/
│   └── cvae_nb_norman_20260420_103000/
│       └── ...
└── vae/
    └── cvae_pbmc3k_20260418_142010/
        └── ...
```

Why timestamped run names: lets you kick off many runs (different
hyperparameters) without collisions, and sorts chronologically by name.

## Relationship to `output/` and `data/`

All three top-level artifact directories follow the same **tracked-README,
ignored-contents** pattern:

| Dir | Role | Typical contents |
|-----|------|------------------|
| [`data/`](../data/) | Input datasets | scRNA-seq `.h5ad`, reference annotations |
| `runs/` (here) | Per-training-run artifacts | Log, checkpoints, end-of-training samples |
| [`output/`](../output/) | Cross-run derived analyses | Benchmark tables, comparison figures, paper plots |

A useful mental model: **data → runs → output → docs**. Data comes in, runs
produce artifacts, output summarizes and compares across runs, and once
stable, results graduate to [`docs/applications/`](../docs/applications/) or
[`docs/products/`](../docs/products/).

## Workflow

See [`examples/docs/running_experiments.md`](../examples/docs/running_experiments.md)
for the full nohup pattern, monitoring, and pod → local retrieval via
`rsync`.

Quick reference:

```bash
# Launch a training run on the pod
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=runs/flow_matching/mnist_fm_${TS}
mkdir -p "$RUN_DIR"
nohup python examples/flow_matching/01_mnist_flow_matching.py \
    --epochs 300 --wandb --checkpoint-dir "$RUN_DIR" \
    > "$RUN_DIR/training.log" 2>&1 &
echo $! > "$RUN_DIR/pid"

# Pull back to local (from your laptop)
rsync -Pavz \
    --exclude='checkpoint_epoch*.pt' \
    <cluster>:sky_workdir/runs/flow_matching/mnist_fm_${TS}/ \
    ./runs/flow_matching/mnist_fm_${TS}/
```

## Archival / Cleanup

`runs/` directories can accumulate rapidly (each checkpoint is ~500 MB for a
6 M-param model). Suggested cleanup patterns:

- After extracting useful figures into `output/`, consider deleting
  intermediate checkpoints: `rm runs/<topic>/<run>/checkpoint_epoch*.pt`
  while keeping `model_final.pt`, `training.log`, `samples_final.png`
- For runs you'll never reproduce, `rm -rf` the whole run directory
- For runs whose outputs are in the W&B cloud (with artifacts registered),
  you can be more aggressive locally
