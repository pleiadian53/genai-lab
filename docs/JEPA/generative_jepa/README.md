# Generative JEPA

A focused subtopic within the [JEPA documentation series](../README.md):
**how to extend a Joint Embedding Predictive Architecture from a
representation-learning model into a generative one**, with the concrete target
of perturbation response prediction on single-cell data.

Vanilla JEPA predicts *embeddings*, not data. It learns the semantic structure
of how inputs relate — but it cannot emit a sample, define a likelihood, or
quantify uncertainty on its own. This subtopic works through what it takes to
close that gap, surveys the 2025–2026 literature that does so, and commits to a
concrete architecture.

## Documents

| Doc | Contents |
|-----|----------|
| [`architecture_spec.md`](architecture_spec.md) | The design space (four routes to a generative JEPA), the empirical caveat that drives the choice, and a concrete staged architecture — **Generative Perturbation JEPA (GP-JEPA)** — with module/loss/evaluation breakdown. |

## Why this is its own subtopic

The base series ([00–05](../README.md)) treats JEPA as predictive and notes "use
a generative wrapper" without specifying one. That hand-wave hides two
*separate* problems — stochastic prediction and decoding back to data space —
and the right way to solve them depends on recent work (Var-JEPA, Cell-JEPA,
D-JEPA, V-JEPA 2 world models). This subtopic exists to treat that properly and
to host follow-on research as it lands.

## Relationship to the rest of the project

- Builds directly on [04_jepa_perturbseq.md](../04_jepa_perturbseq.md) (the
  predictive baseline).
- Reuses the count-aware decoders in `src/genailab/model/` and the latent
  diffusion infrastructure in `src/genailab/diffusion/`.
- Realizes the JEPA component of the three-stage pipeline described in
  [PROJECT_OVERVIEW.md](../../PROJECT_OVERVIEW.md).
</content>
