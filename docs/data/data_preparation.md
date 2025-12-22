# Data Preparation for Generative Models

This document describes how to obtain and preprocess real-world gene expression datasets for training and evaluating generative models (VAE, diffusion, etc.).

---

## 1. Why Real Data Matters

To objectively compare different generative approaches (VAE vs diffusion, NB vs ZINB, etc.), we need:

- **Real count distributions** with overdispersion and sparsity
- **Meaningful conditions** (tissue, disease, cell type) for conditional generation
- **Held-out test sets** for likelihood-based evaluation

---

## 2. Recommended Datasets

### 2.1 scRNA-seq (Start Here)

| Dataset | Description | Size | Conditions | Link |
|---------|-------------|------|------------|------|
| **PBMC 3k** | Classic starter dataset | ~2,700 cells | Cell type | [10x Genomics](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0) |
| **PBMC 68k** | Larger PBMC dataset | ~68,000 cells | Cell type | [10x Genomics](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) |
| **Tabula Sapiens** | Multi-tissue human atlas | ~500k cells | Tissue, cell type, donor | [Portal](https://tabula-sapiens-portal.ds.czbiohub.org/) |
| **Tabula Muris** | Multi-tissue mouse atlas | ~100k cells | Tissue, cell type | [Portal](https://tabula-muris.ds.czbiohub.org/) |

> **Note**: For UMI-based scRNA-seq, NB is often sufficient. Add ZINB only if NB badly underfits zeros.

### 2.2 Bulk RNA-seq (Later)

| Dataset | Description | Size | Conditions | Link |
|---------|-------------|------|------------|------|
| **GTEx** | Multi-tissue, healthy baseline | ~17k samples | Tissue, sex, age | [GTEx Portal](https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression) |
| **recount3** | Uniformly processed public RNA-seq | Massive | Study-dependent | [recount3](https://rna.recount.bio/) |
| **TCGA** | Cancer transcriptomes | ~11k samples | Cancer type, stage | [GDC Portal](https://portal.gdc.cancer.gov/) |

> **Note**: For bulk RNA-seq, NB is typically the right likelihood; ZINB is rarely necessary.

---

## 3. Preprocessing Scripts

### 3.1 scRNA-seq (Python + Scanpy)

**Script**: `src/genailab/data/sc_preprocess.py`

**What it does**:

1. Loads 10x MTX format or downloads PBMC3k directly
2. Computes QC metrics (n_counts, n_genes, mito %)
3. Filters low-quality cells and genes
4. Computes library size for NB models
5. Saves raw counts as `.h5ad`

**Key principle**: Do NOT normalize or log-transform if using NB/ZINB likelihood.

### 3.2 Bulk RNA-seq (R + recount3)

**Script**: `src/genailab/data/bulk_recount3_preprocess.R`

**What it does**:

1. Downloads uniformly processed counts from recount3
2. Extracts counts matrix and sample metadata
3. Filters lowly-expressed genes
4. Saves as RDS (can convert to CSV for Python)

**Reference**: [recount3 quickstart](https://www.bioconductor.org/packages/devel/bioc/vignettes/recount3/inst/doc/recount3-quickstart.html)

### 3.3 Bulk RNA-seq (Python Alternative)

**Script**: `src/genailab/data/bulk_preprocess.py`

**What it does**:

1. Loads counts from CSV files (exported from R or downloaded from portals)
2. Optionally downloads from GEO using GEOparse
3. Computes library size for NB models
4. Filters lowly-expressed genes
5. Converts to AnnData format (same as scRNA-seq)

**Usage examples**:

```bash
# From CSV files (e.g., exported from R/recount3)
python -m genailab.data.bulk_preprocess csv \
    --counts bulk_counts.csv \
    --metadata bulk_metadata.csv \
    --output bulk.h5ad

# From GEO (requires: pip install GEOparse)
python -m genailab.data.bulk_preprocess geo \
    --geo-id GSE12345 \
    --output bulk.h5ad
```

**Workflow**: Use R/recount3 to download uniformly processed counts, export to CSV, then use Python for ML pipeline

---

## 4. Wiring Conditions for cVAE

Once you have counts and metadata, create a condition table:

### 4.1 Bulk RNA-seq Conditions

| Condition | Type | Example Values |
|-----------|------|----------------|
| `tissue` | Categorical | "liver", "brain", "heart" |
| `disease_status` | Categorical | "healthy", "tumor", "treated" |
| `batch` | Categorical | "batch1", "batch2" |
| `sex` | Categorical | "M", "F" |
| `age` | Continuous | 25, 45, 67 |

### 4.2 scRNA-seq Conditions

| Condition | Type | Example Values |
|-----------|------|----------------|
| `cell_type` | Categorical | "T cell", "B cell", "Monocyte" |
| `tissue` | Categorical | "blood", "lung", "liver" |
| `donor` | Categorical | "donor1", "donor2" |
| `batch` | Categorical | "10x_v2", "10x_v3" |

These become categorical IDs → embedding tables in the cVAE encoder/decoder.

---

## 5. Critical Checklist for NB/ZINB Models

- [ ] **Keep raw counts** in the training tensor (no normalization)
- [ ] **Compute library size** (total counts per sample/cell) as offset or covariate
- [ ] **Start with NB**, upgrade to ZINB only if NB underfits zeros on held-out data
- [ ] **Include batch** as a condition (even if you later want invariance)
- [ ] **Filter genes**: Remove genes expressed in <3 cells/samples
- [ ] **Filter cells/samples**: Remove outliers by QC metrics

---

## 6. Library Size: Why It Matters

Library size (total counts per cell/sample) varies due to technical factors, not biology.

For NB models, the typical parameterization is:

$$
\mu_g = \ell \cdot \exp(\eta_g)
$$

where:

- $\ell$ = library size (or learned size factor)
- $\eta_g$ = what the decoder predicts from $(z, y)$

**How to compute**:

```python
# scRNA-seq (scanpy)
adata.obs["library_size"] = np.array(adata.X.sum(axis=1)).ravel()

# Bulk RNA-seq (pandas)
library_size = counts.sum(axis=0)  # sum over genes
```

---

## 7. Output Format for ML

Both scRNA-seq and bulk RNA-seq should produce:

| File | Contents |
|------|----------|
| `counts.h5ad` or `counts.csv` | Raw count matrix (genes × samples/cells) |
| `metadata.csv` | Sample/cell metadata with conditions |
| `library_size.npy` | Precomputed library sizes |

The `.h5ad` format (AnnData) is preferred because it stores counts, metadata, and gene info together.

---

## References

- [Scanpy tutorials](https://scanpy.readthedocs.io/en/stable/tutorials.html)
- [recount3 quickstart](https://www.bioconductor.org/packages/devel/bioc/vignettes/recount3/inst/doc/recount3-quickstart.html)
- [GTEx Portal](https://gtexportal.org/)
- [10x Genomics Datasets](https://www.10xgenomics.com/datasets)