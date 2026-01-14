# Documentation

Tutorial documents and reference materials for **genai-lab**.

Use these documents alongside the code and notebooks to understand the theory, implementation details, and practical applications of generative models in computational biology.

---

## Quick Navigation

| Topic | Description | Start Here |
|-------|-------------|------------|
| 🆕 [foundation_models/](./foundation_models/) | Foundation model adaptation | [leveraging_foundation_models_v2.md](./foundation_models/leveraging_foundation_models_v2.md) |
| 🆕 [DiT/](./DiT/) | Diffusion Transformers | [README.md](./DiT/README.md) |
| 🆕 [JEPA/](./JEPA/) | Joint Embedding Predictive Architecture | [README.md](./JEPA/README.md) |
| 🆕 [latent_diffusion/](./latent_diffusion/) | Latent diffusion for biology | [README.md](./latent_diffusion/README.md) |
| [DDPM/](./DDPM/) | Denoising Diffusion Probabilistic Models | [README.md](./DDPM/README.md) |
| [flow_matching/](./flow_matching/) | Flow matching & rectified flow | [README.md](./flow_matching/README.md) |
| [VAE/](./VAE/) | Variational Autoencoders | [VAE-01-overview.md](./VAE/VAE-01-overview.md) |
| [beta-VAE/](./beta-VAE/) | Disentanglement | [beta_vae.md](./beta-VAE/beta_vae.md) |
| [score_matching/](./score_matching/) | Score functions & energy models | [README.md](./score_matching/README.md) |
| [EBM/](./EBM/) | Energy-Based Models | [README.md](./EBM/README.md) |
| [datasets/](./datasets/) | Datasets & pipelines | [README.md](./datasets/README.md) |
| [runpods/](./runpods/) | GPU cloud setup (RunPod) | [project_setup_on_new_pod.md](./runpods/project_setup_on_new_pod.md) |
| [incubation/](./incubation/) | Experimental ideas | [README.md](./incubation/README.md) |

---

## Topic Guides

### Foundation Models (New!)

Comprehensive guides for adapting pretrained foundation models:

| Document | Content |
|----------|---------|  
| [leveraging_foundation_models_v2.md](./foundation_models/leveraging_foundation_models_v2.md) | Tutorial on adaptation patterns (LoRA, adapters, freezing) |
| [data_shape_v2.md](./foundation_models/data_shape_v2.md) | Understanding transformer tensor shapes |
| [IMPLEMENTATION_GUIDE.md](./foundation_models/IMPLEMENTATION_GUIDE.md) | Quick reference for using the framework |

### DiT (Diffusion Transformers)

Complete series on Diffusion Transformers for biology:

| Document | Content |
|----------|---------|  
| [README.md](./DiT/README.md) | Series overview and navigation |
| [00_dit_overview.md](./DiT/00_dit_overview.md) | Introduction to DiT |
| [01_dit_foundations.md](./DiT/01_dit_foundations.md) | Architecture and components |
| [02_dit_training.md](./DiT/02_dit_training.md) | Training with rectified flow |
| [03_dit_sampling.md](./DiT/03_dit_sampling.md) | Sampling strategies |
| [open_research_tokenization.md](./DiT/open_research_tokenization.md) | Tokenization for biology |

### JEPA (Joint Embedding Predictive Architecture)

Self-supervised learning for computational biology:

| Document | Content |
|----------|---------|  
| [README.md](./JEPA/README.md) | Series overview and navigation |
| [00_jepa_overview.md](./JEPA/00_jepa_overview.md) | Introduction to JEPA |
| [01_jepa_foundations.md](./JEPA/01_jepa_foundations.md) | Architecture and VICReg |
| [02_jepa_training.md](./JEPA/02_jepa_training.md) | Training strategies |
| [03_jepa_applications.md](./JEPA/03_jepa_applications.md) | Biology applications |
| [04_jepa_perturbseq.md](./JEPA/04_jepa_perturbseq.md) | Complete Perturb-seq implementation |
| [open_research_joint_latent.md](./JEPA/open_research_joint_latent.md) | Joint latent spaces |

### Latent Diffusion

Diffusion in VAE latent space for gene expression:

| Document | Content |
|----------|---------|  
| [README.md](./latent_diffusion/README.md) | Series overview and navigation |
| [00_latent_diffusion_overview.md](./latent_diffusion/00_latent_diffusion_overview.md) | Introduction |
| [01_latent_diffusion_foundations.md](./latent_diffusion/01_latent_diffusion_foundations.md) | VAE + DiT architecture |
| [02_latent_diffusion_training.md](./latent_diffusion/02_latent_diffusion_training.md) | Two-stage training |
| [03_latent_diffusion_applications.md](./latent_diffusion/03_latent_diffusion_applications.md) | Biology applications |
| [04_latent_diffusion_combio.md](./latent_diffusion/04_latent_diffusion_combio.md) | End-to-end implementation |

### DDPM (Denoising Diffusion)

Core diffusion model theory and gene expression architectures:

| Document | Content |
|----------|---------|  
| [02a_diffusion_arch_gene_expression.md](./DDPM/02a_diffusion_arch_gene_expression.md) | Architectures for gene expression |
| [02b_diffusion_arch_qa.md](./DDPM/02b_diffusion_arch_qa.md) | Q&A on conditioning complexity |

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

### Datasets

| Document | Content |
|----------|---------|
| [datasets/README.md](./datasets/README.md) | Datasets overview and index |
| [datasets/gene_expression/](./datasets/gene_expression/) | Gene expression (PBMC, RNA-seq) |
| [datasets/medical_imaging/](./datasets/medical_imaging/) | Medical imaging (Chest X-ray) |
| [datasets/perturbation/](./datasets/perturbation/) | Perturbation data (scPerturb) |

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
| `docs/datasets/` | `src/genailab/data/`, `src/genailab/diffusion/datasets.py` | `examples/02_pbmc3k_cvae_nb.ipynb` |

### Reference Style

When working through a notebook, keep the relevant doc open:

- **Training a VAE?** → Read [VAE-model-training.md](./VAE/VAE-model-training.md)
- **Confused about ELBO?** → Read [VAE-02-elbo.md](./VAE/VAE-02-elbo.md)
- **Preprocessing scRNA-seq?** → Read [PBMC.md](./datasets/gene_expression/PBMC.md)

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
