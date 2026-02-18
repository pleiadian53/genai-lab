# Ideas Under Incubation

This directory contains exploratory architectural proposals and application ideas that are not yet implemented. These documents capture emerging directions for the genai-lab project.

## Contents

| Document | Focus | Status |
|----------|-------|--------|
| [alternative_backbones_for_biology.md](alternative_backbones_for_biology.md) | SSMs, tokenization alternatives, architectures for biology | Conceptual |
| [generative-ai-for-gene-expression-prediction.md](generative-ai-for-gene-expression-prediction.md) | Diffusion/VAE/Flow for gene expression with uncertainty, count data handling | Next after Perturb-seq |
| [numerical_embeddings_and_continuous_values.md](numerical_embeddings_and_continuous_values.md) | Encoding strategies for continuous biological values | Research |

## Graduated Documents

Documents that have been promoted from incubation to active use:

| Original Document | Graduated To | Reason |
|-------------------|--------------|--------|
| `joint_latent_space_and_JEPA.md` | [docs/JEPA/05_joint_latent_spaces.md](../JEPA/05_joint_latent_spaces.md) | Supports active Perturb-seq application; JEPA theory complete |
| `generative-ai-for-perturbation-modeling.md` | [docs/applications/architectural_approaches.md](../applications/architectural_approaches.md) | Foundational for active perturbation prediction application |

## Key Themes

### 1. Beyond Tokenization

Gene expression doesn't naturally tokenize like text or images. We explore:

- **State-vector representations** (no tokens)
- **Latent-space diffusion** (VAE + flow matching)
- **Set-based representations** (permutation-invariant)
- **SSM backbones** (Mamba, S4) for temporal dynamics

### 2. Count Data Handling

Gene expression is counts, not continuous values. Solutions:

- **Latent diffusion + NB/ZINB decoder** (recommended)
- **Log-transform** (simple baseline)
- **Discrete diffusion** (future research)

### 3. Hybrid Predictive-Generative Models

Combine supervised predictors (like GEM-1) with generative wrappers:

- Predictor learns conditional mean
- Generative model learns residual distribution
- Enables uncertainty quantification

### 4. Joint Latent Spaces

Static (bulk RNA-seq) and dynamic (Perturb-seq) data can share the same manifold:

- JEPA for predicting embeddings
- Rectified flow for generation
- Patch-n-Pack for heterogeneous batching

## Research Directions

**Near-term:**

- Latent rectified flow for gene expression
- Set Transformer for expression (permutation-invariant)

**Medium-term:**

- Mamba backbone for perturb-seq
- Hybrid architectures (SSM + Transformer)

**Long-term:**

- When does tokenization help vs hurt?
- Biological priors in architecture design

## Related Documentation

- [ROADMAP.md](../ROADMAP.md) — Project learning progression
- [Rectified Flow Tutorial](../flow_matching/rectifying_flow.md)
- [Diffusion Transformer Tutorial](../diffusion/DiT/diffusion_transformer.md)
