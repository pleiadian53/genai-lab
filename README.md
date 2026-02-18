# genai-lab

**Generative AI for Computational Biology**: Research into foundation models and generative methods for accelerating drug discovery, understanding treatment responses, and enabling in silico biological experimentation.

## Overview

This project investigates generative modeling approaches across computational biology, inspired by emerging platforms such as:

- **Gene Expression**: [Synthesize Bio](https://www.synthesize.bio/) (GEM-1), [Deep Genomics](https://www.deepgenomics.com/) (BigRNA)
- **DNA Sequence**: [Arc Institute](https://arcinstitute.org/tools/evo) (Evo 2), [InstaDeep](https://www.instadeep.com/) (Nucleotide Transformer)
- **Single-Cell**: [Geneformer](https://huggingface.co/ctheodoris/Geneformer), [scGPT](https://github.com/bowang-lab/scGPT)
- **Gene Editing**: [Profluent](https://www.profluent.bio/) (OpenCRISPR)

**Research Goals:**

1. **Learn & Document** state-of-the-art generative architectures (VAE, flows, diffusion, transformers) for biological data - *extensive theory documentation complete*
2. **Implement** end-to-end applications that demonstrate practical value - *current focus: perturbation prediction*
3. **Benchmark** against published methods with reproducible workflows - *in progress*
4. **Integrate** causal inference methods for counterfactual validation - *planned collaboration with [causal-bio-lab](../causal-bio-lab/)*

**Current Stage:** Transitioning from methodology exploration to application consolidation. Priority is completing one flagship application (Perturb-seq perturbation prediction) before expanding to additional use cases.

See [docs/INDUSTRY_LANDSCAPE.md](docs/INDUSTRY_LANDSCAPE.md) for a comprehensive survey of companies and technologies in this space.

## Project Structure

```text
genai-lab/
├── src/genailab/
│   ├── foundation/     # 🆕 Foundation model adaptation framework
│   │   ├── configs/        # Resource-aware model configs (small/medium/large)
│   │   ├── tuning/         # LoRA, adapters, freezing strategies
│   │   ├── conditioning/   # FiLM, cross-attention, CFG (planned)
│   │   └── recipes/        # End-to-end pipelines (planned)
│   ├── data/           # Data loading, transforms, preprocessing
│   │   ├── paths.py        # Standardized data path management
│   │   ├── sc_preprocess.py    # scRNA-seq preprocessing (Scanpy)
│   │   └── bulk_preprocess.py  # Bulk RNA-seq preprocessing
│   ├── model/          # Encoders, decoders, VAE, diffusion architectures
│   │   ├── vae.py          # CVAE, CVAE_NB, CVAE_ZINB
│   │   ├── encoders.py     # ConditionEncoder, etc.
│   │   ├── decoders.py     # Gaussian, NB, ZINB decoders
│   │   └── diffusion/      # Diffusion models (DDPM, score networks)
│   ├── objectives/     # Loss functions, regularizers
│   │   └── losses.py       # ELBO, NB, ZINB losses
│   ├── eval/           # Metrics, diagnostics, plotting
│   ├── workflows/      # Training, simulation, benchmarking
│   └── utils/          # Config, reproducibility
├── docs/               # Theory documents and derivations
│   ├── foundation_models/  # 🆕 Foundation model adaptation
│   ├── DiT/            # 🆕 Diffusion Transformers
│   ├── JEPA/           # 🆕 Joint Embedding Predictive Architecture
│   ├── latent_diffusion/   # 🆕 Latent diffusion for biology
│   ├── DDPM/           # Denoising Diffusion Probabilistic Models
│   ├── VAE/            # VAE theory and derivations
│   ├── EBM/            # Energy-based models
│   ├── score_matching/ # Score matching and energy functions
│   ├── flow_matching/  # Flow matching & rectified flow
│   └── datasets/       # Data preparation guides
├── notebooks/          # Educational tutorials (interactive learning)
│   ├── foundation_models/  # 🆕 Foundation adaptation tutorials
│   ├── diffusion/      # Diffusion models tutorials
│   ├── vae/            # VAE tutorials
│   └── foundations/    # Mathematical foundations
├── examples/           # Production scripts (real-world applications)
│   ├── perturbation/   # Drug response, perturbation prediction
│   └── utils/          # Helper modules for examples
├── scripts/            # Training scripts with CLI
│   └── diffusion/      # Diffusion model training scripts
├── data/               # Local data storage (gitignored)
├── tests/
└── environment.yml     # Conda environment specification
```

## Documentation & Learning Resources

### Theory Documents (`docs/`)

Detailed theory, derivations, and mathematical foundations:

| Topic | Description | Start Here |
|-------|-------------|------------|
| 🆕 [foundation_models](docs/foundation_models/) | Foundation model adaptation (LoRA, adapters, freezing) | [leveraging_foundation_models_v2.md](docs/foundation_models/leveraging_foundation_models_v2.md) |
| 🆕 [DiT](docs/DiT/) | Diffusion Transformers (architecture, training, sampling) | [README.md](docs/DiT/README.md) |
| 🆕 [JEPA](docs/JEPA/) | Joint Embedding Predictive Architecture | [README.md](docs/JEPA/README.md) |
| 🆕 [latent_diffusion](docs/latent_diffusion/) | Latent diffusion with NB/ZINB decoders | [README.md](docs/latent_diffusion/README.md) |
| [DDPM](docs/DDPM/) | Denoising Diffusion Probabilistic Models | [README.md](docs/DDPM/README.md) |
| [VAE](docs/VAE/) | Variational Autoencoders (ELBO, inference, training) | [VAE-01-overview.md](docs/VAE/VAE-01-overview.md) |
| [beta-VAE](docs/beta-VAE/) | VAE with disentanglement (β parameter) | [beta_vae.md](docs/beta-VAE/beta_vae.md) |
| [EBM](docs/EBM/) | Energy-Based Models (Boltzmann, partition functions) | [README.md](docs/EBM/README.md) |
| [score_matching](docs/score_matching/) | Score functions, Fisher vs Stein scores | [README.md](docs/score_matching/README.md) |
| [flow_matching](docs/flow_matching/) | Flow matching & rectified flow | [README.md](docs/flow_matching/README.md) |
| [datasets](docs/datasets/) | Datasets & preprocessing pipelines | [README.md](docs/datasets/README.md) |
| [incubation](docs/incubation/) | Ideas under development | [README.md](docs/incubation/README.md) |

### Application Guides (`docs/applications/`)

Application-specific architectures and implementation strategies:

| Application | Description | Status |
|-------------|-------------|--------|
| [perturbation_prediction](docs/applications/perturbation_prediction.md) | End-to-end guide for Perturb-seq modeling | 🎯 Active |
| Gene expression prediction | Hybrid predictive-generative models | 📋 Planned |
| Synthetic data generation | Biology-aware generative pipelines | 📋 Planned |

### Ideas Under Incubation (`docs/incubation/`)

Early-stage architectural explorations not yet integrated into active applications:

| Document | Focus | Status |
|----------|-------|--------|
| [alternative_backbones_for_biology.md](docs/incubation/alternative_backbones_for_biology.md) | SSMs, non-tokenization approaches | Conceptual |
| [generative-ai-for-gene-expression-prediction.md](docs/incubation/generative-ai-for-gene-expression-prediction.md) | Hybrid models for gene expression | Next after Perturb-seq |
| [numerical_embeddings_and_continuous_values.md](docs/incubation/numerical_embeddings_and_continuous_values.md) | Encoding strategies for continuous biological values | Research |

### Interactive Tutorials (`notebooks/`)

Educational Jupyter notebooks for hands-on learning:

| Topic | Description | Start Here |
|-------|-------------|------------|
| 🆕 [foundation_models](notebooks/foundation_models/) | Foundation model adaptation (LoRA, adapters, resource management) | [README.md](notebooks/foundation_models/README.md) |
| [diffusion](notebooks/diffusion/) | Diffusion models (DDPM, score-based, flow matching) | [01_ddpm_basics.ipynb](notebooks/diffusion/01_ddpm_basics.ipynb) |
| [vae](notebooks/vae/) | VAE tutorials (coming soon) | - |
| [foundations](notebooks/foundations/) | Mathematical foundations (coming soon) | - |

See [notebooks/README.md](notebooks/README.md) for learning paths and progression.

### Production Examples (`examples/`)

Ready-to-use Python scripts for real-world applications:

- `01_bulk_cvae.ipynb` — Train CVAE on bulk RNA-seq
- `02_pbmc3k_cvae_nb.ipynb` — Train CVAE with NB decoder on scRNA-seq
- `perturbation/` — Drug response and perturbation prediction (coming soon)

**How to use:**

- **Learning**: Start with `notebooks/` for interactive tutorials
- **Theory**: Reference `docs/` for detailed derivations
- **Application**: Use `examples/` for production workflows
- Follow the [ROADMAP](docs/ROADMAP.md) for structured progression

## Installation

### Using mamba + poetry (recommended)

```bash
# Create conda environment
mamba create -n genailab python=3.11 -y
mamba activate genailab

# Install poetry if not available
pip install poetry

# Install package in editable mode
poetry install

# Optional: install bio dependencies (scanpy, anndata)
poetry install --with bio

# Optional: install dev dependencies
poetry install --with dev
```

### Quick start

```bash
# Verify installation
python -c "import genailab; print(genailab.__version__)"

# Run toy training (once implemented)
genailab-train --config configs/cvae_toy.yaml
```

## Project Status

### Mature Components (Production-Ready ✅)

**VAE Family** - Battle-tested implementations:
- [x] Core CVAE implementation with condition encoding
- [x] Gaussian decoder (MSE reconstruction)
- [x] Negative Binomial decoder for count data (`CVAE_NB`)
- [x] Zero-Inflated Negative Binomial decoder (`CVAE_ZINB`)
- [x] ELBO loss with KL annealing support
- [x] Comprehensive documentation (VAE-01 through VAE-09)
- [x] Unit tests for all model variants

**Data Pipeline** - Operational:
- [x] Standardized data path management (`genailab.data.paths`)
- [x] scRNA-seq preprocessing with Scanpy
- [x] Bulk RNA-seq preprocessing (Python + R/recount3)
- [x] Environment setup (conda/mamba + Poetry)
- [x] Data preparation documentation

### Theory Complete, Implementation Validated 🔬

**Diffusion Infrastructure** - Core diffusion mechanics working:
- [x] Forward/reverse diffusion process (VP-SDE, VE-SDE)
- [x] Score networks (MLP, TabularScoreNetwork, UNet2D, UNet3D)
- [x] Medical imaging diffusion (synthetic X-rays) - proof of concept validated
- [x] Training scripts with configurable model sizes
- [x] RunPod setup documentation for GPU training
- [x] Comprehensive DDPM documentation series
- [x] Score matching theory (denoising score matching, energy functions)

### Active Development 🎯

**Application: Perturbation Prediction (Perturb-seq)** - Current focus:
- [x] JEPA architecture documented for Perturb-seq ([04_jepa_perturbseq.md](docs/JEPA/04_jepa_perturbseq.md))
- [x] Perturbation modeling strategy documented ([docs/applications/](docs/applications/))
- [ ] JEPA implementation for Perturb-seq
- [ ] Latent diffusion for uncertainty quantification
- [ ] Benchmark against scGen/CPA/scPPDM on Norman et al. 2019 dataset

### Research Prototypes (Theory Complete, Implementation Pending 📝)

**Advanced Architectures** - Fully documented, awaiting implementation:
- [x] DiT (Diffusion Transformers) - complete documentation series
- [x] Latent Diffusion with NB/ZINB decoders - complete documentation
- [x] Flow Matching & Rectified Flow - theory documented
- [ ] DiT implementation for gene expression
- [ ] Flow matching implementation

**Foundation Model Adaptation Framework** - Partially implemented:
- [x] Resource-aware model configurations (small/medium/large)
- [x] Auto-detection of hardware (M1 Mac, RunPod, Cloud)
- [x] LoRA (Low-Rank Adaptation) implementation
- [ ] Adapters and freezing strategies
- [ ] Conditioning modules (FiLM, cross-attention, CFG)
- [ ] Tutorial notebooks for each adaptation pattern
- [ ] End-to-end recipes for gene expression tasks

### Planned (After Current Focus) 🔮

**Additional Applications:**
- [ ] Gene expression prediction (GTEx, harmonized bulk RNA-seq)
- [ ] Synthetic biological dataset generation with validation pipeline
- [ ] Conditional generation with classifier-free guidance

**Causal & Counterfactual Methods:**
- [ ] Counterfactual generation pipeline
- [ ] Deconfounding / SCM-flavored latent model
- [ ] Causal regularization via invariance
- [ ] Integration with [causal-bio-lab](../causal-bio-lab/) for causal validation

## Industry Landscape

Companies and platforms pioneering generative AI for drug discovery and biological research:

### Gene Expression & Multi-Omics Foundation Models

| Company | Focus | Key Technology |
|---------|-------|----------------|
| [Synthesize Bio](https://www.synthesize.bio/) | Gene expression generation | GEM-1 foundation model |
| [Ochre Bio](https://www.ochre-bio.com/) | Liver disease, RNA therapeutics | Functional genomics + AI |
| [Deep Genomics](https://www.deepgenomics.com/) | RNA biology & therapeutics | BigRNA (~2B params) |
| [Helical](https://www.helical.ai/) | DNA/RNA foundation models | Helix-mRNA, open-source platform |
| [Noetik](https://www.noetik.ai/) | Cancer biology | OCTO model for treatment prediction |

### Protein & Structure-Based Discovery

| Company | Focus | Key Technology |
|---------|-------|----------------|
| [Isomorphic Labs](https://www.isomorphiclabs.com/) | Drug discovery (DeepMind spin-off) | AlphaFold 3 |
| [EvolutionaryScale](https://www.evolutionaryscale.ai/) | Protein design | ESM3 generative model |
| [Generate:Biomedicines](https://generatebiomedicines.com/) | Protein therapeutics | Generative Biology™ platform |
| [Chai Discovery](https://www.chaidiscovery.com/) | Molecular structure | Chai-1/2 (antibody design) |
| [Recursion](https://www.recursion.com/) | Phenomics + drug discovery | Phenom-Beta, BioHive-2 |

### Clinical & Treatment Response

| Company | Focus | Key Technology |
|---------|-------|----------------|
| [Insilico Medicine](https://insilico.com/) | End-to-end drug discovery | Pharma.AI, Precious3GPT |
| [Tempus](https://www.tempus.com/) | Precision medicine | AI-driven clinical insights |
| [Owkin](https://www.owkin.com/) | Clinical trials, pathology | Federated learning |
| [Retro Biosciences](https://www.retro.bio/) | Cellular reprogramming | GPT-4b micro (with OpenAI) |

### Other Notable Players

- **BioMap** — xTrimo (210B params, multi-modal)
- **Ginkgo Bioworks** — Synthetic biology + Google Cloud partnership
- **Bioptimus** — H-Optimus-0 pathology foundation model
- **Atomic AI** — RNA structure (ATOM-1, PARSE platform)
- **Enveda Biosciences** — PRISM for small molecule discovery

## References

### Academic

- [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) — Transfer learning for single-cell biology
- [scVI](https://www.nature.com/articles/s41592-018-0229-2) — Probabilistic modeling of scRNA-seq
- [CPA](https://www.embopress.org/doi/full/10.15252/msb.202211517) — Compositional Perturbation Autoencoder

### Industry

- [Synthesize Bio Blog](https://www.synthesize.bio/blog)
- [17 Companies Pioneering AI Foundation Models in Pharma](https://www.biopharmatrend.com/business-intelligence/14-companies-pioneering-ai-foundation-models-in-pharma-and-biotech/)
- [NVIDIA BioNeMo Platform](https://blogs.nvidia.com/blog/drug-discovery-bionemo-generative-ai/)

## Related Projects

### [causal-bio-lab](../causal-bio-lab/) — Causal AI/ML for Computational Biology

**Complementary Focus:** While `genai-lab` focuses on **modeling data-generating processes** through generative models, `causal-bio-lab` focuses on **uncovering causal structures** and **estimating causal effects** from observational and interventional data.

**Synergy:**

- **Generative models** (VAE, diffusion) can learn rich representations but may capture spurious correlations
- **Causal methods** (probabilistic graphical models, causal discovery, structural equations) ensure models capture true mechanisms, not just statistical patterns
- **Together:** Causal generative models combine the best of both worlds—realistic simulation with causal guarantees

**Key Integration Points:**

1. **Causal representation learning:** Learn disentangled latent spaces that respect causal structure (causal VAEs, identifiable VAEs)
2. **Causal discovery for architecture:** Use learned causal graphs to constrain generative model structure
3. **Counterfactual validation:** Use causal inference methods (do-calculus, structural equations) to validate generated predictions
4. **Causal regularization:** Apply invariance principles and interventional consistency losses for better generalization

**Example Workflow:**

```text
1. Train a CVAE on gene expression data (genai-lab)
2. Discover causal gene regulatory network (causal-bio-lab)
3. Constrain VAE latent space to respect causal structure
4. Generate counterfactual perturbation responses with causal guarantees
5. Estimate treatment effects using both generative and causal methods
```

**Why This Matters for Computational Biology:**

- **Drug discovery:** Generate realistic molecular perturbations while ensuring causal mechanisms are preserved
- **Treatment response:** Predict individual-level effects (counterfactuals) with uncertainty quantification
- **Target identification:** Discover causal drivers, not just biomarkers
- **Combination therapy:** Model synergistic effects through causal interaction terms

See `causal-bio-lab` Milestone 0.5 (SCMs) and Milestone D (Causal Representation Learning) for integration work.

## License

MIT
