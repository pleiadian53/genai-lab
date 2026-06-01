# genai-lab — Project Overview

A research codebase exploring **generative AI for computational biology** —
translating state-of-the-art generative architectures into practical
applications for drug discovery, treatment-response simulation, and
in-silico biological experimentation.

This document expands on the project [README](../README.md) with a fuller
view of mission, scope, organization, and current state. It's meant for
anyone discovering the project — collaborators, reviewers, or readers
following the work.

---

## Mission

The project investigates how modern generative methods — the VAE family,
diffusion and flow models, Joint Embedding Predictive Architectures (JEPA),
and transformers — can address concrete questions in computational biology
that classical methods struggle with:

- **Predicting cellular responses to genetic and chemical perturbations**
- **Generating biologically realistic expression states under specified conditions**
- **Quantifying uncertainty in predictions, not just point estimates**
- **Reasoning counterfactually about cell state changes**

These are problems where the data is high-dimensional, sparse, count-valued,
and small relative to the questions being asked — exactly the regime where
generative modeling with strong inductive biases (count-aware likelihoods,
self-supervised pretraining, latent diffusion) can outperform discriminative
baselines.

## Scientific Scope

The flagship application area is **perturbation response prediction on
single-cell Perturb-seq data**. The benchmark is the Norman et al. 2019
dataset (K562 cells, CRISPR activation, combinatorial perturbations) — the
de facto reference for the field.

The technical approach combines three complementary ideas:

1. **Count-aware variational autoencoders** with negative-binomial decoders
   for the deterministic baseline — fast point estimates of perturbed cell
   states, suitable for sanity checks and as a comparison floor.
2. **Joint Embedding Predictive Architectures (JEPA)** with perturbation
   conditioning for self-supervised prediction in latent space — learning
   what a cell would look like under a new perturbation without
   reconstructing the full gene-count profile. (The phrase "predict
   embeddings, not pixels" comes from JEPA's vision lineage, where the
   observation is an image; here the observation is a raw gene-count vector
   of ~20k mostly-zero entries, and the same logic applies — predicting in
   embedding space avoids spending model capacity fitting dropout noise.)
3. **Latent diffusion** wrapped around the predictive latent for
   uncertainty quantification — sampling diverse plausible perturbed
   states rather than committing to a single prediction.

Each component addresses a different epistemic need: the baseline gives
interpretable means, JEPA captures the structure of how perturbations
deform the latent manifold, and the diffusion layer gives confidence
intervals on counterfactual queries.

## A Note on JEPA

JEPA (Joint Embedding Predictive Architecture, from the LeCun-school work
including I-JEPA and V-JEPA) is technically a *predictive /
representation-learning* architecture rather than a generative one — it
predicts the *embedding* of a target from the embedding of a context,
without reconstructing data or defining an explicit likelihood.

For genai-lab, JEPA serves as the latent-space predictor, and the project's
contribution is wrapping its output with a generative head (latent
diffusion) so the combined system both predicts and quantifies uncertainty.
This pairing — strong predictive latent + generative sampling on top — is
the core architectural bet of the project.

## Current Stage

The project is **transitioning from broad methodology exploration to
focused application consolidation.** A previous phase produced
documentation, theory derivations, and partial implementations across many
architectures; the current phase consolidates one complete vertical
(perturbation prediction, benchmarked end-to-end) before expanding.

A five-tier status system distinguishes documentation from implementation
from validation throughout the codebase:

| Tier | Symbol | Meaning |
|------|--------|---------|
| Mature | ✅ | Theory + implementation + validated on a realistic dataset |
| Validated | 🔬 | Theory + implementation, validated on toy/benchmark data |
| Active | 🎯 | Current development focus |
| Prototype | 📝 | Theory complete, implementation pending or partial |
| Planned | 🔮 | Designed, not yet started |

Tier assignments live alongside the components they describe; the project
README has the current snapshot.

## How the Project Is Organized

### A maturity ladder

Work flows through three deliberately gated stages:

```
Stage 1 — Use case under development
   examples/<topic>/ + notebooks/<topic>/
   Scripts run; results may be preliminary

       ↓ promote when documented end-to-end with ≥1 concrete result

Stage 2 — Application
   docs/applications/<topic>.md
   Methodology write-up; reproducible by a reader

       ↓ promote when API stabilizes, tests cover inference,
         and a published baseline is matched

Stage 3 — Product
   docs/products/<name>/  +  src/genailab/applications/<name>/
   Deployable: stable API, versioned checkpoints, documented limitations
```

A *product* has a user contract; an *application* has only a methodology
claim. Promotion is a one-way gate; demotion is allowed and expected when
the underlying assumptions change. The full promotion criteria live in
[`docs/products/README.md`](products/README.md).

### Repository structure

```
src/genailab/         Library package (importable)
  foundation/         Foundation-model adaptation (LoRA, configs, hardware detection)
  data/               AnnData loaders, preprocessing
  model/              Encoders, decoders, VAE variants
  diffusion/          VP-SDE, VE-SDE, score networks, training/sampling
  flow_matching/      Rectified flow library
  objectives/         Loss functions (ELBO, NB, ZINB, CFM)
  eval/               Metrics, diagnostics
  applications/       Flagship application code (stable APIs)
  utils/              Config, reproducibility helpers

examples/<topic>/     Production-style scripts per topic
notebooks/<topic>/    Tutorials and exploration, parallel to examples/
ops/                  GPU cluster provisioning (SkyPilot + RunPod)
docs/                 Public documentation (MkDocs)
  applications/         Methodology write-ups
  products/             Mature, deployable applications
  VAE/ DDPM/ JEPA/ …    Methodology-first theory docs by topic
data/                 Datasets (not tracked in git)
runs/                 Per-training-run artifacts (not tracked)
output/               Cross-run analyses and figures (not tracked)
```

Each topic in `examples/` typically has a parallel directory in
`notebooks/` — a well-developed topic has both: a notebook walking through
intuition and a script running the benchmark at scale.

### Tech stack

- **Python 3.10–3.12**, PyTorch only (no TensorFlow)
- **Single-cell**: scanpy, anndata, GEOparse
- **Reference methods** (optional, per task): scvi-tools, scgen, cpa-tools,
  diffusers
- **Experiment tracking**: Weights & Biases
- **GPU provisioning**: SkyPilot + RunPod for real training; local CPU for
  development and small tutorials
- **Environment**: mamba / conda; environment name `genailab`

Domain conventions deserve specific call-out:

- **Raw counts are preserved** for negative-binomial and zero-inflated
  negative-binomial decoders; normalization is only applied to copies for
  descriptive analyses
- **Library size is treated as a covariate**, not a preprocessing step —
  computed on the full filtered gene set and passed explicitly to NB-family
  decoders
- **CPU is the local default** for correctness; CUDA is reserved for real
  training on provisioned cloud GPUs

## Sibling Projects

genai-lab cross-pollinates with related research codebases in the same
broader bio-AI portfolio:

- **agentic-spliceai** — splice-site prediction with agentic validation
  workflows. Several engineering patterns (GPU provisioning via SkyPilot,
  milestone-gated example scripts, session-summary discipline) originated
  there and were ported here.
- **combio-lab** — biomolecular design, including protein structure
  prediction and embeddings. Shares the examples/notebooks parallel
  convention.
- **causal-bio-lab** — causal inference for biological systems. Planned
  integration point: once the flagship application lands, causal methods
  can be used to validate counterfactual predictions against
  intervention-derived ground truth.

## Where to Read Next

| Topic | File |
|-------|------|
| Project entry point | [`README.md`](../README.md) |
| Flagship application (perturbation prediction) | [`docs/applications/perturbation_prediction.md`](applications/perturbation_prediction.md) |
| Industry landscape | [`docs/INDUSTRY_LANDSCAPE.md`](INDUSTRY_LANDSCAPE.md) |
| Maturity ladder + product criteria | [`docs/products/README.md`](products/README.md) |
| GPU workflow | [`ops/README.md`](../ops/README.md) |
| Experiment running + tracking | [`examples/docs/running_experiments.md`](../examples/docs/running_experiments.md), [`examples/docs/experiment_tracking.md`](../examples/docs/experiment_tracking.md) |
| Theory by topic | [`docs/VAE/`](VAE/), [`docs/DDPM/`](DDPM/), [`docs/JEPA/`](JEPA/), [`docs/flow_matching/`](flow_matching/) |
| Dataset background (flagship) | [`notebooks/perturbation/docs/norman_2019_dataset_tutorial.md`](../notebooks/perturbation/docs/norman_2019_dataset_tutorial.md) |
