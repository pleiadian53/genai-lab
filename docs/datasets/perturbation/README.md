# Perturbation Datasets

Datasets for training perturbation prediction models (scPPDM, JEPA) on CRISPR screening and Perturb-seq data.

---

## Overview

Perturbation datasets capture cellular responses to genetic interventions (CRISPR knockouts, knockdowns, overexpression). These are essential for:

- **scPPDM**: Single-cell Perturbation Prediction via Diffusion Models
- **JEPA**: Joint Embedding Predictive Architecture for perturbation response
- **Counterfactual generation**: Predicting "what if" scenarios

---

## Key Datasets

| Dataset | Cells | Perturbations | Cell Type | Document |
|---------|-------|---------------|-----------|----------|
| scPerturb | Multiple | Multiple | Various | [scperturb.md](scperturb.md) |
| Replogle 2022 | ~2.5M | >5,000 | K562 | [scperturb.md](scperturb.md) |
| Norman 2019 | ~100k | ~300 (combinatorial) | K562 | [scperturb.md](scperturb.md) |
| Adamson 2016 | ~10k | ~100 | K562 | [scperturb.md](scperturb.md) |

---

## Data Structure

Perturb-seq data typically includes:

```
AnnData object:
├── X: Expression matrix (cells × genes)
├── obs: Cell metadata
│   ├── perturbation: Gene targeted
│   ├── is_control: Boolean (NT control)
│   └── cell_type, batch, etc.
└── var: Gene metadata
    └── gene_name, highly_variable, etc.
```

### Key Fields

- **Control cells**: Non-targeting (NT) guide, baseline expression
- **Perturbed cells**: Expression after intervention
- **Perturbation label**: Which gene was targeted

---

## Use Cases in genai-lab

### 1. scPPDM (Diffusion-based)

Train diffusion model to predict post-perturbation expression:

```
Control expression + Perturbation embedding → Diffusion → Perturbed expression
```

### 2. JEPA (Embedding-based)

Predict perturbation effects in latent space:

```
Control latent + Perturbation → Predictor → Perturbed latent
```

### 3. Counterfactual Generation

Generate "what if" scenarios:

- What if gene X was knocked out in this cell?
- What is the distribution of possible outcomes?

---

## Documents

- [scperturb.md](scperturb.md) — scPerturb harmonized collection
- [perturb_seq_guide.md](perturb_seq_guide.md) — General Perturb-seq data handling

---

## Related Documentation

- [Generative AI for Perturbation Modeling](../../incubation/generative-ai-for-perturbation-modeling.md)
- [Joint Latent Spaces and JEPA](../../incubation/joint_latent_space_and_JEPA.md)
- [Alternative Backbones for Biology](../../incubation/alternative_backbones_for_biology.md)
