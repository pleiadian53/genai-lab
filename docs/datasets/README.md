# Datasets

This directory documents datasets used in **genai-lab** for training and evaluating generative models, along with their preprocessing pipelines and related code.

---

## Overview

| Category | Datasets | Use Cases |
|----------|----------|-----------|
| [Gene Expression](gene_expression/) | PBMC 3k/68k, bulk RNA-seq | VAE training, latent diffusion |
| [Medical Imaging](medical_imaging/) | Chest X-ray (synthetic & real) | Diffusion models, DiT |
| [Perturbation](perturbation/) | scPerturb, Replogle, Norman | scPPDM, JEPA, perturbation prediction |

---

## Gene Expression Datasets

Single-cell and bulk RNA-seq datasets for generative modeling.

| Document | Description |
|----------|-------------|
| [PBMC.md](gene_expression/PBMC.md) | PBMC 3k/68k dataset guide |
| [data_preparation.md](gene_expression/data_preparation.md) | RNA-seq preprocessing workflows |

**Related code:**

- `src/genailab/data/` — Data loading utilities
- `src/genailab/data/sc_dataset.py` — Single-cell dataset classes
- `notebooks/diffusion/04_gene_expression_diffusion/` — Gene expression diffusion demo

---

## Medical Imaging Datasets

Datasets for training diffusion models on medical images.

| Document | Description |
|----------|-------------|
| [chest_xray.md](medical_imaging/chest_xray.md) | Chest X-ray datasets (synthetic & real) |

**Related code:**

- `src/genailab/diffusion/datasets.py` — `SyntheticXRayDataset`, `ChestXRayDataset`
- `notebooks/diffusion/03_medical_imaging_diffusion/` — Medical imaging diffusion demo

---

## Perturbation Datasets

Perturb-seq and CRISPR screening datasets for perturbation prediction models.

| Document | Description |
|----------|-------------|
| [scperturb.md](perturbation/scperturb.md) | scPerturb harmonized collection |
| [perturb_seq_guide.md](perturbation/perturb_seq_guide.md) | General Perturb-seq data handling |

**Target applications:**

- **scPPDM**: Single-cell Perturbation Prediction via Diffusion Models
- **JEPA**: Joint Embedding Predictive Architecture for perturbation response
- **Counterfactual generation**: "What if" perturbation scenarios

---

## Data Pipeline Pattern

Each dataset follows a consistent pipeline:

```
Raw Data → Preprocessing → PyTorch Dataset → DataLoader → Model
              ↓
         Normalization
         Quality Control
         Train/Val/Test Split
```

**Key considerations:**

1. **Gene expression**: Log-transform, HVG selection, NB/ZINB for counts
2. **Medical imaging**: Resize, normalize to [-1, 1], augmentation
3. **Perturbation**: Control vs treated pairing, batch correction

---

## Adding New Datasets

When documenting a new dataset:

1. Create a markdown file in the appropriate subdirectory
2. Include:
   - **Source**: Where to download, licensing
   - **Description**: What the data contains
   - **Preprocessing**: Required transformations
   - **Code**: Related modules and notebooks
   - **Example usage**: Code snippets
3. Update this README with a link

---

## Related Documentation

- [ROADMAP.md](../ROADMAP.md) — Learning progression
- [Latent Diffusion + NB/ZINB](../incubation/generative-ai-for-gene-expression-prediction.md) — Count data handling
- [scPPDM concepts](../incubation/generative-ai-for-perturbation-modeling.md) — Perturbation modeling
