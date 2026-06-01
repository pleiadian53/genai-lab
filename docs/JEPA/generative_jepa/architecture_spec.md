# Generative Perturbation JEPA (GP-JEPA) — Architecture Spec

A concrete design for turning a Joint Embedding Predictive Architecture from a
representation learner into a *generative* model of single-cell perturbation
response. This document states the problem precisely, surveys the routes the
recent literature offers, explains the empirical finding that constrains the
choice, and commits to a staged architecture grounded in this project's existing
components.

**Prerequisites**: [JEPA foundations](../01_jepa_foundations.md),
[Perturb-seq application](../04_jepa_perturbseq.md). Familiarity with
negative-binomial decoders (see the [VAE docs](../../VAE/)) helps.

---

## 1. Why vanilla JEPA isn't generative: two separate gaps

It is tempting to say "JEPA isn't generative because it has no decoder." That is
only half the story. There are **two independent** missing pieces, and a usable
generative model must close both.

| Gap | What's missing in JEPA | What generation requires |
|-----|------------------------|--------------------------|
| **G1 — Stochastic prediction** | The predictor `z_pred = predictor(z_context, condition)` is a *deterministic point estimate*. One perturbation maps to exactly one latent. | A *distribution* over outcomes — one perturbation should map to a population of plausible perturbed cell states, because real cells respond heterogeneously. |
| **G2 — No decoder to data space** | JEPA predicts an *embedding* and trains it (with VICReg) to match a target encoder. There is no map from latent back to gene counts. | A decoder `z → x` that emits an actual expression vector. |

A model that solves only G2 (add a decoder to a deterministic predictor) gives
one expression profile per perturbation — a point estimate, not a generative
model. A model that solves only G1 (a distribution over latents) can sample
embeddings but never produce data. **Both** are required.

This framing also clarifies the project's current plan ("wrap a latent-diffusion
head on the JEPA latent"): it is one way to solve G1 and G2 with a single
bolted-on module. It works, but it is not the only option, and the recent
literature offers alternatives that integrate more cleanly.

---

## 2. The design space — four routes

| Route | Idea | Solves | Trade-off |
|-------|------|--------|-----------|
| **A — Latent decoder head** | Attach a negative-binomial (NB/ZINB) decoder to the predicted latent; make the predictor or decoder stochastic. | G1 (if stochastic) + G2 | Lowest friction, reuses count-aware decoders. Risk: collapses toward a conditional VAE, so JEPA's added value must be demonstrated, not assumed. |
| **B — Variational JEPA** | Reformulate JEPA variationally: a posterior over latent predictions, a learnable conditional prior, and a decoder that reconstructs in *representation* space. (cf. Var-JEPA.) | G1 + G2 | Cleanest theory; the predictor *becomes* the conditional prior. Extra variational machinery and predictive-vs-generative loss balancing. |
| **C — Representation-conditioned diffusion** | Keep JEPA as a representation engine; train a separate diffusion model *conditioned on* the JEPA latent to emit data. (cf. D-JEPA.) | G1 (diffusion noise) + G2 (diffusion decode) | Most modular; either half is swappable. Heaviest — two models to train and tune. |
| **D — World-model planning** | Freeze the JEPA encoder, train an *action-conditioned* predictor ("if perturbation *a*, then latent *z′*"), and *plan* by minimizing a goal-conditioned energy over the space of perturbations. (cf. V-JEPA 2-AC.) | Neither G1 nor G2 directly — generative in the *decision* sense: it generates *interventions*. | Highest leverage for screening / drug discovery; does not by itself produce expression profiles, so it sits on top of A/B/C. |

Routes A–C produce **data** (expression profiles); Route D produces
**decisions** (which perturbation to apply to reach a target state). They are
complementary, not competing.

---

## 3. The empirical finding that constrains the choice

A naive reading of the JEPA literature suggests "predict in latent space, skip
the decoder, win." For single-cell perturbation prediction specifically, the
evidence is more nuanced.

Recent work applying JEPA to single-cell transcriptomics (Cell-JEPA, 2026)
reports a strong result on *representation* quality — large gains in zero-shot
cell-type transfer over reconstruction-based foundation models — but a pointed
limitation on *perturbation* tasks:

> Within a single cell line, the approach improves **absolute-state
> reconstruction but not effect-size estimation.**

This matters because the standard perturbation benchmark measures **effect
size**: the correlation between predicted and true differential expression on
the most-affected genes. A model can have an excellent latent representation and
still mis-estimate the *magnitude* of a perturbation's effect.

**Consequence for this design**: a generative head that maps back to count space
(Route A or B) is not optional polish — it is the mechanism by which effect sizes
are recovered and calibrated. Route D alone (latent-similarity planning) does not
address it. This is why the architecture below makes a count-space decoder
central rather than peripheral.

---

## 4. The chosen architecture — GP-JEPA

A three-stage model that uses each route where it is strongest:
intra-cell JEPA for a dropout-robust encoder, an action-conditioned predictor as
the world-model core, and a variational + NB-decoder head to close G1 and G2 and
recover effect sizes.

```
                       ┌──────────────────── Stage A: encoder pretraining ───────────────────┐
   raw counts x  ──►   masked-gene context  ──► f_θ (encoder) ──► z      target: f_θ̄(x) [EMA]
                                                                  └── VICReg(z, target) ──► dropout-robust f_θ

                       ┌──────────────── Stage B: perturbation predictor (world-model core) ──┐
   control cell x_b ─► f_θ ─► z_b
   perturbation p   ─► e_θ ─► z_p ─►  g_φ(z_b, z_p)  ──►  μ_pred, log σ²_pred   (posterior q)
                                       learnable prior π(z | z_b, z_p)

                       ┌──────────────── Stage C: generative head (closes G1 + G2) ───────────┐
   sample  ẑ ~ q(z | z_b, z_p)                       ← G1: stochastic prediction
   library size ℓ ─►  decoder d_ψ(ẑ, ℓ) ──► NB(μ, θ) ──► counts x̂        ← G2: decode to data
```

### 4.1 Stage A — self-supervised encoder pretraining (intra-cell JEPA)

- **Goal**: a cell encoder `f_θ` whose latent is robust to single-cell dropout
  (>90% zeros), where reconstruction objectives waste capacity on measurement
  artifacts.
- **Mechanism**: mask a random subset of a cell's genes; predict the *embedding*
  of the held-out genes from the visible ones. Target embeddings come from an
  EMA "teacher" copy `f_θ̄`. VICReg (variance–invariance–covariance) prevents
  collapse.
- **Why JEPA here**: this is the regime where latent prediction clearly beats
  reconstruction (the zero-shot transfer result). Trains on *unlabeled* cells, so
  it can use far more data than the perturbation-labelled subset.
- **Reuses**: the gene-expression encoder pattern from
  [01_jepa_foundations.md](../01_jepa_foundations.md).

### 4.2 Stage B — perturbation-conditioned predictor (the world-model core)

- **Goal**: model the *action* of a perturbation on cell state —
  `z_perturbed ≈ g_φ(z_baseline, z_perturbation)`.
- **Mechanism**: context = encoded control cell `z_b`; condition = perturbation
  embedding `z_p` (learned per-gene embeddings; combinations composed, e.g.
  `A+B`). The predictor `g_φ` is a conditional transformer. This is structurally
  identical to an action-conditioned world-model predictor ("if I apply action
  *a* in state *s*, the next state is *s′*") — the perturbation *is* the action.
- **Key change vs. the baseline predictor**: `g_φ` outputs the **parameters of a
  posterior** `q(z | z_b, z_p)` (mean and log-variance), not a single `z_pred`.
  This is what makes G1 solvable in Stage C.

### 4.3 Stage C — generative head (closes G1 and G2)

This is the part that makes the model generative. Two sub-components:

- **G1 — variational prediction (Route B).** Draw `ẑ ~ q(z | z_b, z_p)`. A
  learnable conditional prior `π(z | z_b, z_p)` is regularized against the
  posterior (KL term). Sampling `ẑ` repeatedly yields a *population* of perturbed
  latents — the cell-to-cell heterogeneity of a real perturbation response.
- **G2 — count-space decoding (Route A).** Decode `ẑ` with a negative-binomial
  decoder `d_ψ(ẑ, ℓ)` conditioned on library size `ℓ`, following this project's
  decoder convention (softmax over genes → scale by library size → NB mean;
  per-gene dispersion `θ = exp(log θ)`, kept contiguous). This produces actual
  counts **and** recovers the effect-size magnitude the latent-only model misses.
- **Swappable variant (Route C).** For richer multimodal uncertainty, replace the
  Gaussian posterior in G1 with a **latent diffusion** process over `z`
  conditioned on `(z_b, z_p)`, then decode as above. This preserves the project's
  original "latent diffusion for uncertainty quantification" bet while keeping the
  encoder and predictor unchanged. Start with the variational head (simpler,
  cheaper); escalate to diffusion if posterior calibration is insufficient.

### 4.4 Application layer — world-model screening (Route D)

Once the predictor is calibrated, the same `g_φ` supports *planning*: given a
desired phenotype embedding `z_goal` (e.g. a healthy or differentiated state),
search the space of perturbations to minimize a goal-conditioned energy

```
E(p) = ‖ g_φ(z_baseline, e_θ(p)) − z_goal ‖
```

using a sampling-based optimizer (e.g. the Cross-Entropy Method), exactly as
action-conditioned video world models plan toward a goal image. This turns
[`virtual_screen_perturbations()`](../04_jepa_perturbseq.md) from an exhaustive
ranking of a fixed candidate list into an *optimization* over perturbation /
combination space — in-silico screening as planning. It is "generative" in the
decision sense: it generates interventions, not expression vectors.

---

## 5. Training objective

A single objective with four terms, trained in stages (A, then B+C jointly):

```
L = L_predict        (latent prediction: posterior mean vs. target encoder)
  + λ_vic · L_VICReg (anti-collapse, Stages A and B)
  + λ_nb  · L_NB     (negative-binomial reconstruction of counts, Stage C)
  + λ_kl  · L_KL     (posterior vs. learnable conditional prior, Stage C)
```

- `L_predict` and `L_VICReg` carry the JEPA representation quality.
- `L_NB` is what forces effect-size calibration (the Cell-JEPA caveat).
- `L_KL` is what makes the model generative (a samplable prior).
- Balancing `λ_nb` against the latent terms is the main tuning lever; too small
  and effect sizes are uncalibrated, too large and the model degenerates into a
  plain conditional NB-VAE (losing the JEPA pretraining benefit).

---

## 6. Evaluation — report both axes

The central methodological discipline: **never report representation quality
alone.** A model can ace latent metrics and still fail the benchmark that
matters. Report both:

| Axis | Metrics | What it catches |
|------|---------|-----------------|
| **Representation** | Zero-shot cell-type transfer (AvgBIO), latent neighborhood structure, held-out-perturbation embedding similarity | Whether the encoder learned biology — JEPA's strength. |
| **Effect size** | Pearson / R² between predicted and true differential expression on top-K DE genes; whole-transcriptome correlation | Whether the model got the *magnitude* of the response right — the standard benchmark, and JEPA's known weak spot. |
| **Generative quality** | Calibration of predicted cell-to-cell variance vs. observed; coverage of posterior intervals on held-out perturbations | Whether the uncertainty is real, not decorative. |

Generalization splits should follow the standard protocols (unseen
perturbations; unseen *combinations*) so the world-model claim is tested on
genetic-interaction structure, not just memorized singletons.

---

## 7. Module mapping

| Component | Lives in / reuses |
|-----------|-------------------|
| Encoder `f_θ`, predictor `g_φ` | `src/genailab/applications/perturbation/models/` (new), built on [01_jepa_foundations.md](../01_jepa_foundations.md) patterns |
| NB/ZINB decoder `d_ψ` | `src/genailab/model/decoders.py` (existing) |
| Latent diffusion variant | `src/genailab/diffusion/` (existing VP/VE-SDE + score networks) |
| Perturbation embedding `e_θ` | `ConditionEncoder` pattern (ConditionSpec → Embedding → MLP) |
| VICReg, NB, KL losses | `src/genailab/objectives/` (existing ELBO/NB/ZINB; add VICReg) |
| Data, library size covariate | `src/genailab/applications/perturbation/data/norman.py` (existing) |

---

## 8. Design summary

- JEPA is not generative for **two** reasons — no stochastic prediction (G1) and
  no decoder (G2). Close both, deliberately.
- For single-cell perturbation, latent-space prediction alone improves
  representation but **not** effect-size estimation; a count-space decoder is the
  fix, so make it central.
- **GP-JEPA**: intra-cell JEPA pretraining (robust encoder) → action-conditioned
  predictor (world-model core) → variational + NB head (generative, effect-size
  calibrated), with latent diffusion as a swappable uncertainty variant and
  energy-minimization planning as the screening application.
- Evaluate on representation **and** effect-size **and** calibration — reporting
  only the first is the trap the literature warns against.

---

## References

- Assran et al. (2023), *I-JEPA* — joint-embedding predictive architecture for images.
- Bardes et al. (2024); Meta AI (2025), *V-JEPA / V-JEPA 2* — video world models;
  action-conditioned prediction and energy-minimization planning.
  https://arxiv.org/html/2506.09985v1
- *Cell-JEPA* (2026) — JEPA for single-cell transcriptomics; representation vs.
  effect-size finding. https://arxiv.org/abs/2602.02093
- *Var-JEPA* (2026) — variational formulation bridging predictive and generative
  SSL. https://arxiv.org/pdf/2603.20111
- *D-JEPA / JEPA + diffusion noise* — representation-conditioned generation.
  https://arxiv.org/html/2507.15216v1
- *Causal-JEPA* (2026) — world models via object-level latent interventions.
  https://arxiv.org/pdf/2602.11389
- Norman et al. (2019) — Perturb-seq genetic-interaction manifold (benchmark dataset).
- Lotfollahi et al. (2019), *scGen*; Roohani et al. (2023), *GEARS* — perturbation
  prediction baselines.
</content>
