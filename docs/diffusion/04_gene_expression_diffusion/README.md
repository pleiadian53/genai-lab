# Diffusion Models for Gene Expression Data

This notebook extends the SDE-based diffusion framework to generate **realistic gene expression data** — a key capability for drug discovery and computational biology.

## Motivation

Companies like **Synthesize Bio** (GEM-1), **Insilico Medicine** (Precious3GPT), and **scGPT** are building generative models for gene expression. Applications include:

- **Drug target discovery** — Generate expression profiles under hypothetical perturbations
- **Clinical trial acceleration** — In-silico patient simulation
- **Data augmentation** — Generate rare cell types or disease states

## Key Concepts

### Why Latent Diffusion?

| Challenge | Direct Diffusion | Latent Diffusion |
|-----------|------------------|------------------|
| Dimensionality | 2,000-20,000 genes | 32-128 latent dims |
| Training speed | Slow | Fast |
| Structure | May miss correlations | VAE captures structure |
| Conditioning | Complex | Natural via embeddings |

### Architecture

```
Gene Expression (n_genes) 
    → VAE Encoder 
    → Latent Space (z_dim) 
    → Diffusion (VP-SDE) 
    → VAE Decoder 
    → Gene Expression (n_genes)
```

## Learning Objectives

1. Understand challenges of applying diffusion to gene expression
2. Implement latent diffusion for high-dimensional biological data
3. Add conditional generation (tissue, disease, cell type)
4. Evaluate with biological metrics (gene correlations, PCA overlap)

## Prerequisites

- `02_sde_formulation/` — SDE basics and score matching
- Understanding of VAEs (variational autoencoders)

## Files

- `03_gene_expression_diffusion.ipynb` — Main tutorial notebook

## Next Steps

After this notebook:
1. Apply to real single-cell data (PBMC3k via `SingleCellDataset`)
2. Add conditional generation for perturbation prediction
3. Connect to scPPDM framework for drug response modeling

## References

- [Synthesize Bio](https://www.synthesize.bio/) — GEM-1 gene expression model
- [scGPT](https://github.com/bowang-lab/scGPT) — Generative pre-trained transformer for single-cell
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) — Latent diffusion for images
