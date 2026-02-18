# Application Guides

This directory contains end-to-end application guides for generative AI in computational biology. Unlike the methodology-focused documentation in the parent directories, these guides are **application-first** and **results-oriented**.

---

## Philosophy

Each application guide follows a consistent structure:

1. **Problem definition**: What biological question are we answering?
2. **Why generative AI**: What do generative models add over discriminative/deterministic approaches?
3. **Architecture choices**: Which methods from our toolkit (VAE, JEPA, diffusion, flow) and why?
4. **Implementation roadmap**: Step-by-step guide with code examples
5. **Evaluation strategy**: Metrics, benchmarks, biological validation
6. **Expected outcomes**: Quantitative results and scientific insights

---

## Active Applications

### 🎯 [Perturbation Prediction](perturbation_prediction.md)

**Goal**: Predict single-cell responses to genetic/chemical perturbations

**Status**: Active development (Week 1-2: Data + VAE Baseline)

**Architecture**: CVAE_NB → JEPA → Latent Diffusion

**Target Dataset**: Norman et al. 2019 Perturb-seq (K562 cells)

**Key Innovation**: Three-stage approach that combines:
- Count-aware modeling (NB decoders)
- Self-supervised prediction (JEPA)
- Uncertainty quantification (diffusion in latent space)

**See**: [perturbation_prediction.md](perturbation_prediction.md)

---

## Planned Applications

### 📋 Gene Expression Prediction

**Goal**: Predict gene expression from metadata with uncertainty quantification

**Why Generative AI**: 
- GEM-1 and similar models learn $\mathbb{E}[x \mid \text{metadata}]$
- We target $p(x \mid \text{metadata})$ for uncertainty quantification

**Proposed Architecture**: Hybrid predictive-generative
- Stage 1: GEM-1-style supervised predictor (learn conditional mean)
- Stage 2: Diffusion on residuals (learn distribution around mean)

**Target Dataset**: GTEx or harmonized bulk RNA-seq

**Status**: Next after Perturbation Prediction

**Related**: [docs/incubation/generative-ai-for-gene-expression-prediction.md](../incubation/generative-ai-for-gene-expression-prediction.md)

### 📋 Synthetic Biological Data Generation

**Goal**: Generate realistic synthetic datasets for augmentation and benchmarking

**Why Generative AI**:
- Data augmentation for rare conditions
- Privacy-preserving data sharing
- Benchmarking computational methods

**Proposed Architecture**: Conditional diffusion with metadata conditioning

**Target Dataset**: CellxGene census or scPerturb

**Status**: After at least one prediction-focused application is complete

---

## Application Selection Criteria

We prioritize applications based on:

1. **Scientific impact**: Does it address a central problem in computational biology?
2. **Clear benchmarks**: Can we compare against published methods?
3. **Leverages strengths**: Does it use our existing infrastructure (VAE, diffusion, JEPA)?
4. **Demonstrates value**: Does generative AI add something discriminative models cannot?

---

## Relationship to Methodology Documentation

| Directory | Focus | Style |
|-----------|-------|-------|
| [docs/VAE/](../VAE/) | VAE theory and derivations | Methodology-first |
| [docs/DDPM/](../DDPM/) | Diffusion model foundations | Methodology-first |
| [docs/JEPA/](../JEPA/) | Self-supervised prediction | Methodology-first |
| **[docs/applications/](.)** | **End-to-end biological applications** | **Application-first** |

**When to use which**:
- Learning VAE theory? → `docs/VAE/`
- Learning JEPA architecture? → `docs/JEPA/`
- Building a perturbation prediction system? → `docs/applications/perturbation_prediction.md`

---

## Contributing New Applications

When adding a new application guide:

1. **Start with a clear problem**: What biological question?
2. **Justify generative AI**: Why not just discriminative/deterministic models?
3. **Choose architectures deliberately**: From our validated toolbox
4. **Include implementation details**: Code examples, not just ideas
5. **Define success metrics**: How do we know it works?
6. **Validate biologically**: Not just computational metrics

**Template structure**:
```markdown
# Application Name

## Executive Summary
## Background: Why Generative AI for [Problem]?
## Architecture Overview
## Implementation Roadmap
  ### Phase 1: ...
  ### Phase 2: ...
  ### Phase 3: ...
## Expected Outcomes
## Beyond the Flagship: Extensions
## References
## Implementation Checklist
```

---

## Status Dashboard

| Application | Stage | Dataset | Next Milestone |
|-------------|-------|---------|----------------|
| **Perturbation Prediction** | 🎯 Active | Norman et al. 2019 | VAE baseline + metrics |
| Gene Expression Prediction | 📋 Planned | GTEx | (after Perturb-seq) |
| Synthetic Data Generation | 📋 Planned | CellxGene | (after Perturb-seq) |

---

**Last Updated**: 2026-01-31
