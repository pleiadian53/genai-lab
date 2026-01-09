# scPerturb: Harmonized Perturbation Datasets

**scPerturb** is a harmonized collection of single-cell perturbation datasets, providing a unified resource for training and benchmarking perturbation prediction models.

---

## Overview

| Property | Value |
|----------|-------|
| Source | [scperturb.org](https://scperturb.org) |
| Paper | Peidli et al. (2024) "scPerturb: Harmonized Single-Cell Perturbation Data" |
| Format | AnnData (h5ad) |
| License | Various (check individual datasets) |

---

## Available Datasets

### Recommended for scPPDM / JEPA

| Dataset | Perturbations | Cells | Cell Type | Modality | Notes |
|---------|---------------|-------|-----------|----------|-------|
| **Replogle 2022 (K562)** | >5,000 genes | ~2.5M | K562 | CRISPRi | Largest, recommended for training |
| **Norman 2019** | ~300 (combinatorial) | ~100k | K562 | CRISPRa | Combinatorial perturbations |
| **Adamson 2016** | ~100 (UPR) | ~10k | K562 | CRISPRi | Focused on UPR pathway |
| **Dixit 2016** | ~24 | ~10k | Dendritic | CRISPRi | Original Perturb-seq paper |

### Additional Datasets

| Dataset | Focus | Size |
|---------|-------|------|
| Frangieh 2021 | Cancer immunotherapy | ~200k cells |
| Papalexi 2021 | ECCITE-seq (protein + RNA) | ~40k cells |
| Gasperini 2019 | Enhancer perturbations | ~200k cells |

---

## Data Access

### Option 1: scPerturb Portal

Download pre-processed h5ad files from [scperturb.org](https://scperturb.org).

### Option 2: Python API

```python
# Install scperturb
pip install scperturb

# Download dataset
import scperturb
adata = scperturb.load_dataset("replogle_2022_k562")
```

### Option 3: Direct Download

```bash
# Replogle 2022 K562 (large, ~5GB)
wget https://scperturb.org/data/replogle_2022_k562.h5ad

# Norman 2019 (smaller, good for testing)
wget https://scperturb.org/data/norman_2019.h5ad
```

---

## Data Structure

All scPerturb datasets follow a consistent schema:

```python
import scanpy as sc

adata = sc.read_h5ad("replogle_2022_k562.h5ad")

# Key fields in adata.obs:
# - perturbation: Target gene name (e.g., "TP53", "non-targeting")
# - is_control: Boolean, True for non-targeting controls
# - cell_type: Cell type annotation
# - perturbation_type: "CRISPRi", "CRISPRa", "CRISPRko"

# Expression matrix
X = adata.X  # (n_cells, n_genes)

# Get control cells
controls = adata[adata.obs['is_control']]

# Get perturbed cells for specific gene
tp53_ko = adata[adata.obs['perturbation'] == 'TP53']
```

---

## Preprocessing for scPPDM

### Standard Pipeline

```python
import scanpy as sc
import numpy as np

# 1. Load data
adata = sc.read_h5ad("replogle_2022_k562.h5ad")

# 2. Quality control
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 3. Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 4. Select highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]

# 5. Split by perturbation
train_perts = [...]  # List of perturbations for training
test_perts = [...]   # Held-out perturbations

train_adata = adata[adata.obs['perturbation'].isin(train_perts + ['non-targeting'])]
test_adata = adata[adata.obs['perturbation'].isin(test_perts + ['non-targeting'])]
```

### Creating Control-Perturbed Pairs

```python
def create_pairs(adata, perturbation):
    """Create (control, perturbed) pairs for training."""
    controls = adata[adata.obs['is_control']].X
    perturbed = adata[adata.obs['perturbation'] == perturbation].X
    
    # Random pairing (or use more sophisticated matching)
    n_pairs = min(len(controls), len(perturbed))
    ctrl_idx = np.random.choice(len(controls), n_pairs, replace=False)
    pert_idx = np.random.choice(len(perturbed), n_pairs, replace=False)
    
    return controls[ctrl_idx], perturbed[pert_idx]
```

---

## Evaluation Metrics

Standard metrics for perturbation prediction:

| Metric | Description |
|--------|-------------|
| **MSE** | Mean squared error on held-out perturbations |
| **Pearson r** | Correlation between predicted and true expression |
| **DEG overlap** | Overlap of differentially expressed genes |
| **Pathway enrichment** | Biological pathway consistency |

---

## Use in genai-lab

### For scPPDM (Diffusion)

```python
# Train diffusion model to predict perturbation effects
# Input: control expression + perturbation embedding
# Output: perturbed expression distribution
```

### For JEPA

```python
# Train predictor in latent space
# Input: control latent + perturbation token
# Output: predicted perturbed latent
```

---

## Recommended Starting Point

For initial experiments, use **Norman 2019**:

- Smaller size (~100k cells)
- Well-characterized perturbations
- Combinatorial effects (interesting for modeling)
- Widely used benchmark

For production training, use **Replogle 2022**:

- Largest dataset (>5,000 perturbations)
- Comprehensive coverage
- High quality

---

## References

- Peidli et al. (2024) "scPerturb: Harmonized Single-Cell Perturbation Data"
- Replogle et al. (2022) "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq"
- Norman et al. (2019) "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes"
- Dixit et al. (2016) "Perturb-Seq: Dissecting Molecular Circuits with Scalable Single-Cell RNA Profiling of Pooled Genetic Screens"

---

## Related Documentation

- [Perturbation Datasets Overview](README.md)
- [Generative AI for Perturbation Modeling](../../incubation/generative-ai-for-perturbation-modeling.md)
- [JEPA for Perturb-seq](../../incubation/joint_latent_space_and_JEPA.md)
