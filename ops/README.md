# ops/ — GPU Cluster Provisioning

Programmatic provisioning of RunPod GPU clusters via [SkyPilot](https://skypilot.co/).
Lets you spin up a pod, stage data, and SSH in with one command — no manual
pod setup, no manual `rsync`, no stale SSH configs.

This sub-project is infrastructure, not genai code. Treat it as a utility that
any topic under `examples/<topic>/` or `notebooks/<topic>/` can invoke when a
task needs a GPU.

---

## When to Use This

| Scenario | Tool |
|----------|------|
| Toy/small tasks that fit on your laptop (PBMC 3k CVAE) | Run locally (CPU or MPS) |
| Anything realistic — Norman 2019 Perturb-seq, JEPA training, diffusion runs | `ops/provision_cluster.py` |
| Multi-GPU distributed training | `ops/provision_cluster.py --gpu a100` or `h100` |

If you're reaching for a pod, you should be reaching for this — not the legacy
manual workflow in `runpods/` (see [Relationship to `runpods/`](#relationship-to-runpods) below).

---

## Prerequisites

1. **SkyPilot + RunPod credentials** installed once:
   ```bash
   pip install "skypilot[runpod]"
   sky check runpod   # should say "enabled"
   ```
2. **A RunPod network volume** named in [`configs/gpu_config.yaml`](configs/gpu_config.yaml)
   (default: `"AI lab extension"`). Rename or create via the RunPod dashboard.
3. **Data organized** as `data/<modality>/<sub-topic>/<dataset>/` at the
   project root (e.g., `data/scrna/perturb_seq/norman_2019/`,
   `data/scrna/pbmc/68k/`, `data/bulk/gtex/`).

---

## Common Workflows

### First time — stage a dataset to the network volume

```bash
# Default: stages data/scrna/perturb_seq/norman_2019/
python ops/provision_cluster.py --stage-data

# Stage a different dataset
python ops/provision_cluster.py --stage-data --data-path scrna/pbmc/68k
python ops/provision_cluster.py --stage-data --data-path bulk/gtex
```

This uploads the local dataset to the network volume once. Subsequent
provisions mount the volume instantly — no re-upload.

### Provision a workspace cluster

```bash
# Default: A40 on RunPod, genailab installed, volume mounted
python ops/provision_cluster.py

# Specific GPU
python ops/provision_cluster.py --gpu a100

# With an extra dependency profile (see gpu_config.yaml)
python ops/provision_cluster.py --model scvi
```

The cluster **stays alive** until you explicitly tear it down. This is
intentional — iterative work on a pod is hundreds of times faster than
re-provisioning for each run.

### SSH and run jobs

```bash
ssh genai-workspace           # cluster name printed by provision_cluster.py
cd /workspace/genai-lab       # workdir synced from your local repo
python examples/perturbation/P2_cvae_nb_baseline.py
```

### Tear down

```bash
python ops/provision_cluster.py --status    # list running clusters
python ops/provision_cluster.py --down      # interactive teardown
python ops/provision_cluster.py --down-all  # nuke everything
```

**Tear down when you're done.** Pods cost money whether or not you're using them.

---

## Configuration

Defaults live in [`configs/gpu_config.yaml`](configs/gpu_config.yaml).
CLI flags override the file. The most common edits:

| Field | What it does | When to edit |
|-------|--------------|--------------|
| `gpu` | GPU type (`a40`, `a100`, `h100`, ...) | Change for more/less VRAM |
| `data_path` | Dataset subpath `<modality>/<sub-topic>/<dataset>` | Change for a different dataset |
| `default_model` | Extra pip deps beyond genailab | Set to `scvi`, `scgen`, etc. |
| `use_volume` | Mount the network volume | Set `false` for one-off jobs |

### GPU options and rough pricing (2026)

| Key | GPU | VRAM | ~$/hr |
|-----|-----|------|-------|
| `rtx4000ada` | RTX 4000 Ada | 20 GB | 0.26 |
| `rtxa5000` | RTX A5000 | 24 GB | 0.27 |
| `l4` | L4 | 24 GB | 0.39 |
| `a40` | A40 (default) | 48 GB | 0.39 |
| `rtx4090` | RTX 4090 | 24 GB | 0.59 |
| `rtx5090` | RTX 5090 | 32 GB | 0.89 |
| `a100` | A100 | 80 GB | 1.64 |
| `h100` | H100 | 80 GB | 3.29 |

Pricing is approximate and varies with RunPod availability. Check `sky show-gpus`
for live rates before launching expensive GPUs.

### Model dependency profiles

Most genailab tasks just need the stock PyTorch image + the genailab package,
so `default_model: none`. Add optional profiles as your work expands:

- **scvi** — [scvi-tools](https://scvi-tools.org) reference single-cell VAE
- **scgen** — perturbation response prediction baseline
- **cpa** — Compositional Perturbation Autoencoder
- **diffusers** — HuggingFace diffusion reference implementations

Invoke with `--model <name>`. Extend profiles by editing `models:` in
[`configs/gpu_config.yaml`](configs/gpu_config.yaml).

---

## How It Works

1. `provision_cluster.py` reads `configs/gpu_config.yaml` + CLI overrides
2. Generates a SkyPilot YAML into `configs/skypilot/generated/` (git-ignored)
3. Invokes `sky launch` — provisions pod, runs setup (`pip install -e .`,
   optional model deps), mounts volume
4. Prints SSH instructions + reminder to tear down

The underlying Python API lives in [`gpu_runner.py`](gpu_runner.py):
`GPU_SPECS`, `InfraConfig`, `build_skypilot_config`, `launch`, `stage_data`.
Import these directly if you want to script a pipeline (see docstring).

---

## Relationship to `runpods/`

This repo historically had a manual workflow under `runpods/`:

- User provisions pod via the RunPod web dashboard
- `runpod_ssh_manager.sh` adds an SSH host entry
- `rsync` data manually

That workflow still works, but `ops/` is the recommended approach going forward
because it is:

- **Programmatic** — no web-dashboard clicking, no hand-written configs
- **Repeatable** — same provision every time, same data staging every time
- **Volume-aware** — stages datasets once, reuses them across runs
- **Model-aware** — optional dependency profiles for reference methods

Keep `runpods/` scripts around for one-off or legacy needs; use `ops/` for
everything new.

---

## Attribution

Adapted from [agentic-spliceai's foundation_models/](https://github.com/pleiadian53/agentic-spliceai)
GPU runner. The pattern (SkyPilot + network volume + YAML-driven infra config)
carries across bio-AI projects. If you end up maintaining the same provisioner
across three or more projects, consider extracting it to a standalone repo.
