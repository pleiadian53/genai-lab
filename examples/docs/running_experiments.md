# Running Experiments on a Remote Pod

How to launch long-running training scripts on a RunPod GPU pod so they
survive SSH disconnects, produce organized outputs, and are easy to
monitor, kill, and pull back.

This document is about *how to run* scripts. For pod lifecycle (provisioning,
staging data, teardown) see [`ops/README.md`](../../ops/README.md). For
experiment tracking and metric logging, see
[`experiment_tracking.md`](experiment_tracking.md) *(TODO: to be added).*

---

## TL;DR — launch a detached run

From the project root on the pod (e.g., `~/sky_workdir`):

```bash
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=runs/flow_matching/mnist_fm_${TS}
mkdir -p "$RUN_DIR"

nohup python examples/flow_matching/01_mnist_flow_matching.py \
    --epochs 300 --batch-size 256 --base-channels 128 --n-steps 100 \
    --checkpoint-dir "$RUN_DIR" \
    > "$RUN_DIR/training.log" 2>&1 &

echo $! > "$RUN_DIR/pid"
echo "Run started (PID $(cat $RUN_DIR/pid))"
```

You can now `exit` the SSH session. The script will keep running until
completion, error, or pod teardown.

---

## Run-directory convention

```
runs/<topic>/<run-name>_<YYYYMMDD_HHMMSS>/
├── training.log       # nohup stdout + stderr
├── pid                # single line, the PID of the training process
├── checkpoint_*.pt    # from --checkpoint-dir
├── samples_*.png      # any artifacts the script saves
└── config.json        # (optional) snapshot of CLI args / hyperparameters
```

Why each piece:

- **`<topic>/` layer** — mirrors the `examples/<topic>/` and `notebooks/<topic>/`
  discipline. Keeps runs browsable by subject area (flow_matching,
  perturbation, diffusion, ...).
- **`<run-name>_<timestamp>` suffix** — lets you launch many runs with
  different hyperparameters without collisions. Sort by name to get
  chronological order.
- **`training.log`** — required. `nohup` + redirect captures everything
  the script prints. This is your primary artifact for post-hoc diagnosis.
- **`pid` file** — required. Lets you reattach management (kill, ps, wait)
  from a new SSH session after the original terminal closes. Without it,
  you'd have to `ps aux | grep` and guess.
- **`checkpoint_*.pt`, `samples_*.png`** — produced by the script via
  `--checkpoint-dir`. Co-locating them with the log means one `rsync` pulls
  everything back for a given run.
- **`config.json`** — optional but recommended. Scripts should dump their
  parsed args (e.g., with `json.dump(vars(args), f, indent=2)`) so you can
  reconstruct a run from its directory alone.

---

## Anatomy of the nohup command

```bash
nohup python <script.py> <args...> > "$RUN_DIR/training.log" 2>&1 &
echo $! > "$RUN_DIR/pid"
```

| Piece | Role |
|-------|------|
| `nohup` | Ignores `SIGHUP` when the shell exits, so the child survives logout |
| `python <script.py> <args...>` | The actual training command |
| `> "$RUN_DIR/training.log"` | Redirect stdout to the log file |
| `2>&1` | Redirect stderr to the same stream — so errors land in the log |
| `&` | Run in background — shell returns immediately |
| `echo $!` | `$!` is the PID of the most recently backgrounded process |
| `> "$RUN_DIR/pid"` | Persist the PID so you can find it later |

### Common mistakes

- **Forgetting `2>&1`** → stderr goes to the terminal and is lost on
  disconnect. Always include it.
- **Putting `&` before the redirects** → shell parses as foreground. Always
  `... > log 2>&1 &` in that order.
- **Omitting the `pid` file** → you'll burn 10 minutes hunting the PID
  after your terminal closes.
- **Running from the wrong directory** → if the script uses relative paths
  (`data/...`, `runs/...`), it must be launched from the project root.
  Check `pwd` before launching.

---

## Monitoring a running job

From any SSH session (including a fresh one):

```bash
RUN_DIR=runs/flow_matching/mnist_fm_20260417_144530   # replace with yours

# Is it still running?
ps -p $(cat $RUN_DIR/pid) && echo RUNNING || echo STOPPED

# Live stream of the log
tail -f $RUN_DIR/training.log

# GPU utilization (separate terminal)
watch -n 2 nvidia-smi

# Disk usage of the run
du -sh $RUN_DIR
```

### Killing a stuck or wrong-config run

```bash
kill $(cat $RUN_DIR/pid)
# If it's truly stuck (rare, usually from dead CUDA contexts):
kill -9 $(cat $RUN_DIR/pid)
```

Always kill via the `pid` file, not `pkill python`. Grepping process names
will match any other Python processes (including Jupyter servers, tooling).

---

## Retrieving results from the pod

From your **local** machine, after the run finishes:

```bash
# Pull the entire topic's runs
rsync -Pavz sky-be66-pleiadian53:sky_workdir/runs/flow_matching/ ./runs/flow_matching/

# Or just a single run
rsync -Pavz sky-be66-pleiadian53:sky_workdir/runs/flow_matching/mnist_fm_20260417_144530/ \
             ./runs/flow_matching/mnist_fm_20260417_144530/
```

The `-P` flag gives you progress + partial transfer resume (helpful for
large checkpoint files over flaky network). The `-z` enables compression,
which helps for log files and config JSONs but gives little for already-
compressed binary checkpoints.

### When to rsync

- **Finished runs** — always pull before teardown
- **In-progress runs** — occasionally, if you want local copies of partial
  checkpoints for resume/inspection
- **Log only (quick check)** — `scp sky-be66-pleiadian53:sky_workdir/runs/<topic>/<run>/training.log .`

### Reminder: tear down when done

```bash
# Locally
python ops/provision_cluster.py --down
```

The pod is billed by the minute while it exists, regardless of whether a
training job is running. If you have finished runs on the pod but still
need the pod alive for iterative work, leave it up. Otherwise tear down.

---

## Patterns for multi-run experiments

### Manual sweep over hyperparameters

```bash
for LR in 1e-3 3e-4 1e-4; do
    for BS in 128 256; do
        TS=$(date +%Y%m%d_%H%M%S)
        RUN_DIR=runs/flow_matching/lr${LR}_bs${BS}_${TS}
        mkdir -p "$RUN_DIR"
        nohup python examples/flow_matching/01_mnist_flow_matching.py \
            --lr $LR --batch-size $BS --epochs 100 \
            --checkpoint-dir "$RUN_DIR" \
            > "$RUN_DIR/training.log" 2>&1 &
        echo $! > "$RUN_DIR/pid"
        sleep 2   # avoid timestamp collisions
    done
done
```

**Caveat**: running many concurrent training jobs on a single GPU will
make them all slower than running them sequentially. For true hyperparameter
sweeps, queue them with a simple script or use a tracking tool like
Weights & Biases Sweeps (see `experiment_tracking.md` — TODO).

### Reproducibility

Always record:
- Git commit SHA at launch time (`git rev-parse HEAD > $RUN_DIR/commit.sha`)
- The exact command (`echo "$@" > $RUN_DIR/command.sh` inside your script's
  argparse wrapper, or just log the argparse dict)
- Random seed(s) used
- Data version (e.g., dataset checksum or symlink target)

This is what separates a "run you can reason about" from a "run you have
to reproduce from scratch to compare against."

---

## Related

- [`ops/README.md`](../../ops/README.md) — pod provisioning, SkyPilot,
  network volumes
- [`examples/README.md`](../README.md) — examples directory convention
- [`experiment_tracking.md`](experiment_tracking.md) — *(to be added)*
  MLflow / W&B setup for metric logging and sweep management
