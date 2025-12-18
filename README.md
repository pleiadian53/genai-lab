# genai-lab

**Generative AI for Computational Biology**: Research into foundation models and generative methods for accelerating drug discovery, understanding treatment responses, and enabling in silico biological experimentation.

## Overview

This project investigates generative modeling approaches across computational biology, inspired by emerging platforms such as:

- **Gene Expression**: [Synthesize Bio](https://www.synthesize.bio/) (GEM-1), [Deep Genomics](https://www.deepgenomics.com/) (BigRNA)
- **DNA Sequence**: [Arc Institute](https://arcinstitute.org/tools/evo) (Evo 2), [InstaDeep](https://www.instadeep.com/) (Nucleotide Transformer)
- **Single-Cell**: [Geneformer](https://huggingface.co/ctheodoris/Geneformer), [scGPT](https://github.com/bowang-lab/scGPT)
- **Gene Editing**: [Profluent](https://www.profluent.bio/) (OpenCRISPR)

**Research Goals:**

1. **Investigate** state-of-the-art generative architectures (VAE, flows, diffusion, transformers) for biological sequences and multi-omics data
2. **Develop** reusable, modular components for conditional generation and counterfactual simulation
3. **Explore** causal inference methods for predicting treatment responses and perturbation effects
4. **Contribute** to the growing field of generative biology with reproducible implementations and benchmarks

See [docs/INDUSTRY_LANDSCAPE.md](docs/INDUSTRY_LANDSCAPE.md) for a comprehensive survey of companies and technologies in this space.

## Project Structure

```text
genai-lab/
├── src/genailab/
│   ├── data/           # Data loading, transforms, batch handling
│   ├── model/          # Encoders, decoders, VAE, diffusion
│   ├── objectives/     # Loss functions, regularizers
│   ├── eval/           # Metrics, diagnostics, plotting
│   ├── workflows/      # Training, simulation, benchmarking
│   └── utils/          # Config, reproducibility
├── tests/
├── examples/
└── dev/                # Private notes, brainstorming (not shared)
```

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

## Milestones

### Milestone A: Bulk Conditional VAE

- [ ] Implement cVAE with tissue/disease/batch conditioning
- [ ] Train on toy synthetic data
- [ ] Evaluate: DE agreement, pathway concordance, batch leakage

### Milestone B: scRNA Conditional NB-VAE

- [ ] Negative Binomial likelihood for count data
- [ ] Cell type + donor conditioning
- [ ] Pseudobulk bridging evaluation

### Milestone C: Counterfactual & Causal

- [ ] Counterfactual generation pipeline
- [ ] Deconfounding / SCM-flavored latent model
- [ ] Causal regularization via invariance

### Milestone D: Diffusion Models

- [ ] Latent diffusion for expression
- [ ] Conditional score-based generation

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

## License

MIT
