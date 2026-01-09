# Gene Expression Datasets

Datasets for training generative models on gene expression data (single-cell and bulk RNA-seq).

---

## Available Datasets

| Dataset | Type | Cells/Samples | Genes | Document |
|---------|------|---------------|-------|----------|
| PBMC 3k | scRNA-seq | ~2,700 | ~13,000 | [PBMC.md](PBMC.md) |
| PBMC 68k | scRNA-seq | ~68,000 | ~13,000 | [PBMC.md](PBMC.md) |

---

## Key Considerations for Gene Expression

### Count Data Challenge

Gene expression data consists of **counts**, not continuous values. This requires special handling:

1. **Preprocessing**: `log1p` transform, normalization
2. **Model output**: NB/ZINB decoder (not MSE reconstruction)
3. **Diffusion**: Run in latent space, not raw counts

See [Latent Diffusion + NB/ZINB](../../incubation/generative-ai-for-gene-expression-prediction.md) for details.

### Typical Pipeline

```python
# 1. Load and preprocess
adata = sc.read_h5ad("pbmc3k.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# 2. Create PyTorch dataset
from genailab.data.sc_dataset import AnnDataDataset
dataset = AnnDataDataset(adata, use_hvg=True)

# 3. Train VAE with NB decoder
from genailab.model.vae import GeneVAE_NB
vae = GeneVAE_NB(n_genes=2000, latent_dim=128)
```

---

## Related Code

| Module | Purpose |
|--------|---------|
| `src/genailab/data/sc_dataset.py` | AnnData → PyTorch Dataset |
| `src/genailab/model/decoders.py` | NB/ZINB decoders |
| `src/genailab/objectives/losses.py` | `nb_loss`, `zinb_loss`, `elbo_loss_nb` |

## Related Notebooks

- `notebooks/diffusion/04_gene_expression_diffusion/` — Latent diffusion for gene expression
- `examples/01_bulk_cvae.ipynb` — CVAE on bulk RNA-seq
- `examples/02_pbmc3k_cvae_nb.ipynb` — CVAE with NB decoder on PBMC

---

## Documents

- [PBMC.md](PBMC.md) — PBMC 3k/68k dataset guide
- [data_preparation.md](data_preparation.md) — General RNA-seq preprocessing
