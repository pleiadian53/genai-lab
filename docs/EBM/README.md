# Energy-Based Models (EBM) Documentation

This folder contains documentation on Energy-Based Models, covering the mathematical foundations and the computational challenges that motivate modern training techniques.

## Reading Order

The documents are designed to be read in sequence, building from foundational concepts to more advanced topics:

### Foundations

1. **[Energy Function Normalization](EBM-energy-function-normalization.md)**  
   Proves that the energy-based probability formulation $p_\theta(x) = \exp(-E_\theta(x))/Z_\theta$ is a valid normalized probability density.

2. **[MLE Gradient Derivation](EBM-mle-gradient-derivation.md)**  
   Derives the gradient of the log-likelihood for EBMs, revealing the intractable expectation $\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(x')]$ that makes MLE computationally challenging. This is the "villain origin story."

3. **[Stein vs Fisher Score](EBM-stein-vs-fisher-score.md)**  
   Clarifies the distinction between the Stein score ($\nabla_x \log p$) and Fisher score ($\nabla_\theta \log p$)—two different "scores" used in different contexts.

4. **[Score Matching Objective Derivation](EBM-score-matching-objective.md)**  
   Proves the integration-by-parts trick that eliminates the unknown $p_D$ from the score matching objective, yielding the tractable trace-of-Jacobian form.

5. **[Fisher Score Matching Derivation](EBM-score-matching-FSM-analogue.md)**  
   The parameter-space analogue: proves how integration-by-parts eliminates the intractable $\nabla_\theta \log p(x|\theta)$ for simulation-based inference.

### Training Methods

6. **[Score Matching (detailed)](../score_matching/SM-core-objective.md)** — Full treatment of the score matching objective.

7. **[Fisher Score Matching](../score_matching/SM-fisher-score-matching.md)** — Parameter-space analogue for simulation-based inference.

### Coming Soon

- **Contrastive Divergence** — Approximate MCMC for tractable training
- **Noise-Contrastive Estimation** — Reframing EBM training as classification
- **Denoising Score Matching** — Practical variant avoiding the trace term

## Key Concepts

| Concept | Symbol | Description |
|---------|--------|-------------|
| Energy function | $E_\theta(x)$ | Maps data to scalar "energy" (lower = more probable) |
| Partition function | $Z_\theta$ | Normalizing constant $\int \exp(-E_\theta(x)) dx$ |
| Score function | $\nabla_x \log p(x)$ | Gradient of log-density w.r.t. data |

## Connection to Other Topics

- **VAE**: See [`../VAE/`](../VAE/) — VAEs avoid the partition function problem by using tractable encoder/decoder distributions.
- **Score Matching**: See [`../score_matching/`](../score_matching/) — Directly estimates the score function without computing $Z_\theta$.
- **Diffusion Models**: Build on score matching to learn $\nabla_x \log p_t(x)$ across noise levels.
