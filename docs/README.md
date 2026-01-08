# Documentation

Tutorial documents and reference materials for **genai-lab**.

Use these documents alongside the code and notebooks to understand the theory, implementation details, and practical applications of generative models in computational biology.

---

## Quick Navigation

| Topic | Description | Start Here |
|-------|-------------|------------|
| [VAE/](./VAE/) | Variational Autoencoders | [VAE-01-overview.md](./VAE/VAE-01-overview.md) |
| [beta-VAE/](./beta-VAE/) | Disentanglement | [beta_vae.md](./beta-VAE/beta_vae.md) |
| [score_matching/](./score_matching/) | Score functions & energy models | [README.md](./score_matching/README.md) |
| [EBM/](./EBM/) | Energy-Based Models | [README.md](./EBM/README.md) |
| [data/](./data/) | Data preparation guides | [data_preparation.md](./data/data_preparation.md) |
| [runpods/](./runpods/) | GPU cloud setup (RunPod) | [project_setup_on_new_pod.md](./runpods/project_setup_on_new_pod.md) |
| [incubation/](./incubation/) | Experimental ideas (JEPA, etc.) | - |

---

## Topic Guides

### VAE (Variational Autoencoders)

The VAE series provides a complete treatment from theory to implementation:

| Document | Content |
|----------|---------|
| [VAE-01-overview.md](./VAE/VAE-01-overview.md) | Introduction and motivation |
| [VAE-02-elbo.md](./VAE/VAE-02-elbo.md) | ELBO derivation |
| [VAE-03-inference.md](./VAE/VAE-03-inference.md) | Variational inference |
| [VAE-for-prediction.md](./VAE/VAE-for-prediction.md) | Using VAEs for downstream tasks |
| [VAE-model-training.md](./VAE/VAE-model-training.md) | Training diagnostics & posterior collapse |

### Score Matching & Energy-Based Models

| Document | Content |
|----------|---------|
| [score_matching/README.md](./score_matching/README.md) | Score matching overview |
| [EBM/README.md](./EBM/README.md) | Energy functions and Boltzmann distributions |
| [EBM/EBM-stein-vs-fisher-score.md](./EBM/EBM-stein-vs-fisher-score.md) | Stein vs Fisher score comparison |

### Data Preparation

| Document | Content |
|----------|---------|
| [data/data_preparation.md](./data/data_preparation.md) | RNA-seq preprocessing workflows |
| [data/PBMC.md](./data/PBMC.md) | PBMC 3k/68k dataset guide |

---

## How to Use These Documents

### Learning Path

Follow the [ROADMAP.md](./ROADMAP.md) for a structured progression:

1. **Stage 1**: Start with VAE documents (VAE-01 through VAE-09)
2. **Stage 2**: Data preparation guides
3. **Stage 3**: Score matching and energy-based models
4. **Stage 4+**: Diffusion, causal inference (coming soon)

### Alongside Code

Each topic folder corresponds to implementations in `src/genailab/`:

| Docs Folder | Code Location | Example Notebook |
|-------------|---------------|------------------|
| `docs/VAE/` | `src/genailab/model/vae.py` | `examples/01_bulk_cvae.ipynb` |
| `docs/data/` | `src/genailab/data/` | `examples/02_pbmc3k_cvae_nb.ipynb` |

### Reference Style

When working through a notebook, keep the relevant doc open:

- **Training a VAE?** → Read [VAE-model-training.md](./VAE/VAE-model-training.md)
- **Confused about ELBO?** → Read [VAE-02-elbo.md](./VAE/VAE-02-elbo.md)
- **Preprocessing scRNA-seq?** → Read [PBMC.md](./data/PBMC.md)

---

## Project-Level Documents

| Document | Description |
|----------|-------------|
| [ROADMAP.md](./ROADMAP.md) | Learning progression and milestones |
| [INDUSTRY_LANDSCAPE.md](./INDUSTRY_LANDSCAPE.md) | Survey of companies in generative biology |

---

## Contributing

When adding new documents:

- Place topic-specific docs in the appropriate subfolder (e.g., `VAE/`, `EBM/`)
- Update this README with a link
- Follow the naming convention: `TOPIC-subtopic.md` or `descriptive-name.md`
