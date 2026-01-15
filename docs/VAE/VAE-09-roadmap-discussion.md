# Roadmap Discussion: Adapting GenAI for Gene Expression

This document discusses how to adapt the [ROADMAP.md](../ROADMAP.md) for gene expression data (bulk RNA-seq and scRNA-seq).

The roadmap is strong as a "generative AI curriculum" and mostly workable for gene expression. However, a few stages and metrics need **domain-specific rewiring** to avoid optimizing the wrong thing.

## What's Already a Great Fit

### VAE → cVAE

This is exactly the right foundation *if* you treat gene expression as **counts** (NB/ZINB) rather than "pixels" (Gaussian/MSE).

- **Bulk RNA-seq**: NB is usually the right default; ZINB rarely needed
- **scRNA-seq**: NB is often sufficient for UMI data; ZINB can help for extreme sparsity, but it's not automatically "better"

### Score Matching → DDPM

Good next chapter *if* you pick the right representation. Directly diffusing raw counts is awkward; practical pipelines diffuse:

- **Log-normalized** expression, or
- **Learned latent** space (latent diffusion)

### JEPA / World Models

Especially relevant for biology use cases:

- Perturbation prediction (action = drug/KO)
- Trajectory modeling (action = time)
- Counterfactuals

Keeping JEPA/world models in later stages is sensible.

## What to Adjust for Gene Expression

### Put scVI-Style Likelihoods + Library Size into Stage 1–2

For real gene expression work, the *first* serious milestone should be:

- **NB decoder** for counts (bulk + scRNA)
- Explicit handling of **library size / sequencing depth** (as offset or covariate)
- ZINB only after diagnostics show NB underfits zeros

This is the difference between "VAE toy demo" and "biology-grade model."

### Replace Vision Metrics with Biology Metrics

FID/IS aren't natural for gene expression (they rely on pretrained vision feature extractors). Better metrics:

| Category | Metric |
|----------|--------|
| **Likelihood** | Held-out NB/ZINB log-likelihood (or ELBO) |
| **Distribution** | Gene-wise mean/variance + zero rate matching (per condition) |
| **Structure** | Condition-separation (do generated samples preserve tissue/disease structure?) |
| **Utility** | Downstream classifier trained on real+synthetic → tested on real |

### IWAE: Only if You Hit Posterior Collapse

IWAE is a great learning milestone, but for expression you'll get more value from:

- KL warmup / free bits
- Decoder likelihood correctness (NB)
- Conditioning hygiene

IWAE becomes useful when studying inference quality, but it's not the highest ROI "next step" unless you see issues.

### Flow Matching: After Deciding Data Space vs Latent Space

Flow matching works well *if* you work in:

- Continuous normalized expression space, or
- Latent space from a trained encoder

Tie it explicitly to the representation choice.

## A Two-Track Roadmap for Gene Expression

### Track A: Count-Faithful Representation Learning

1. cVAE with **NB** decoder (bulk + scRNA)
2. Add conditions (tissue/disease/batch) + counterfactual swap
3. β-VAE *only if* you want disentangled residual factors (with good diagnostics)

### Track B: High-Fidelity Generation

1. Learn a good continuous representation (normalized or latent)
2. Score matching / diffusion in that space
3. Conditional sampling (classifier-free guidance for metadata)

### Unification

Both tracks converge for:

- Perturbation response (world model)
- JEPA-style predictive objectives

This preserves the "VAE → score matching → diffusion → JEPA/world models" arc, but makes it biology-native.

## The Key Decision: What Are You Modeling?

For your first real dataset, choose one:

| Representation | Likelihood | Pros | Cons |
|----------------|------------|------|------|
| **Raw counts** | NB/ZINB | Biologically faithful | Harder to model |
| **Log-normalized** | Gaussian | Easier diffusion/flow | Loses count structure |
| **Learned latent** | Gaussian | Best of both worlds | Requires good encoder first |

**Recommendation**: Start with **raw counts** — it forces you to confront the actual generative problem (library size, overdispersion, sparsity, batch, confounding).

## Guardrails for Raw Count Modeling

### Use NB First, Not ZINB

Your MVP should be NB:

$$
x_g \sim \text{NB}(\mu_g, \alpha_g)
$$

ZINB adds an extra head ($\pi$) and can soak up modeling mistakes ("everything becomes dropout"). Upgrade to ZINB only if NB fails clear diagnostics.

### Pick Real but Manageable Datasets

Datasets should be:

- Public, well-described
- Not enormous
- Have clean metadata (tissue/disease/batch)
- Used in prior work (for sanity-checking)

### Minimum Data Hygiene for Raw Counts

If you skip these, NB models will look worse than they are:

- **Gene filtering**: Remove genes expressed in ~0 cells/samples (or keep HVGs)
- **Library size factor** (must-have): Total counts per sample/cell
  
  Typical NB parameterization:

  $$
  \mu_g = \ell \cdot \exp(\eta_g)
  $$

  where $\ell$ is library size and $\eta_g$ is what the decoder predicts from $(z, y)$.

- **Batch covariate**: Include batch in $y$ if present (even if you later want invariance)

### Evaluation Metrics for Real Data

Forget FID. For counts, track:

**Likelihood-Fit Diagnostics**

- Held-out NB log-likelihood / ELBO
- Gene-wise mean/variance vs real (per condition)
- Zero rate vs real (per gene; per condition)

**Structure Diagnostics**

- Train a probe on inferred $z$ to predict batch/tissue (detect leakage/confounding)
- Latent collapse monitoring: average KL, active dimensions

**Usefulness Diagnostics**

- Downstream classifier trained on synthetic + real, tested on real
- DE signature preservation: effect size correlation (real vs generated)

## Where Different Methods Shine

| Method | Best For | Caveats |
|--------|----------|---------||
| **cVAE (NB)** | Controlled generation + counterfactual swaps; latents for world-modeling | — |
| **β-VAE** | Disentangled residual factors | Can hurt reconstruction/log-likelihood |
| **Diffusion/Score** | High-fidelity generation | Awkward on discrete counts; use latent space |

## Practical Staged Plan (Raw Counts)

1. **NB cVAE** on a real dataset with a single clean condition (tissue only OR disease only)
2. Add **library size modeling** explicitly and confirm fit improves
3. Add **batch** conditioning and test counterfactual consistency
4. Test **ZINB** only if NB underfits zeros (check held-out likelihood + zero-rate calibration)
5. Add **β** (β-VAE) only for specific goals: disentangled factors, stable latents

## Bulk vs scRNA: Which First?

For your first real experiment:

| Data Type | Pros | Cons |
|-----------|------|------|
| **Bulk RNA-seq** | Cleaner per sample; simpler | Fewer samples; harder to train deep models |
| **scRNA-seq** | More data points; tests sparsity handling | More nuisance variation |

**Recommendation**: Start with **scRNA-seq (PBMC 3k)** because:

1. **Smaller, faster iteration** — 3k cells vs thousands of samples
2. **Well-documented** — extensively used in tutorials (scanpy, scVI)
3. **Clear ground truth** — known cell types for validation
4. **Directly tests NB/ZINB** — sparsity is real
5. **Preprocessing script ready** — `src/genailab/data/sc_preprocess.py`

## Recommended Datasets

### scRNA-seq (Start Here)

| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| **PBMC 3k** | Classic starter dataset | ~3k cells | [10x Genomics](https://www.10xgenomics.com/resources/datasets) |
| **Tabula Sapiens** | Multi-tissue human atlas | ~500k cells | [Tabula Sapiens](https://tabula-sapiens-portal.ds.czbiohub.org/) |

### Bulk RNA-seq (Later)

| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| **GTEx** | Multi-tissue, healthy baseline | ~17k samples | [GTEx Portal](https://gtexportal.org/) |
| **recount3** | Uniformly processed public RNA-seq | Massive | [recount3](https://rna.recount.bio/) |

> **Note**: For bulk, NB is typically sufficient. For UMI scRNA, NB is often enough; add ZINB only if NB badly underfits zeros.

## References

- [ROADMAP.md](../ROADMAP.md) — Original GenAI roadmap
- [VAE-07-NB-ZINB.md](VAE-07-NB-ZINB.md) — NB vs ZINB likelihood choice
- [VAE-08-NB-likelihood.md](VAE-08-NB-likelihood.md) — NB log-likelihood derivation
- Lopez et al. (2018) — scVI: Deep generative modeling for single-cell transcriptomics
- Eraslan et al. (2019) — DCA: Single-cell RNA-seq denoising using a deep count autoencoder
