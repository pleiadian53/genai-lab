"""Perturbation response prediction — the flagship application.

Three-stage stack for predicting perturbed cell states from baseline
expression + perturbation identity:

  1. Deterministic baseline — CVAE_NB / CVAE_ZINB with perturbation
     conditioning. Fast, serves as sanity check.
  2. Self-supervised prediction — JEPA in latent space, VICReg to prevent
     collapse.
  3. Uncertainty quantification — latent diffusion wrapper for counterfactual
     sampling with confidence intervals.

Flagship dataset: Norman et al. 2019 Perturb-seq (K562, CRISPRa, combinatorial
perturbations). See ``docs/applications/perturbation_prediction.md`` and
``notebooks/perturbation/docs/norman_2019_dataset_tutorial.md``.
"""
