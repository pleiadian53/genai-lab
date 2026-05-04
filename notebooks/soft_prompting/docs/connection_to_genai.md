# Soft Prompting — Connection to genai-lab

## Why this directory exists

`notebooks/soft_prompting/` covers **soft prompting** as a parameter-efficient
fine-tuning (PEFT) technique. The current notebook uses HyenaDNA as a
substrate, but the technique — not the specific foundation model — is the
subject.

Soft prompting prepends a small number of *learned* token embeddings to a
frozen foundation model's input. Only the prompt embeddings are trained.
Trainable parameters scale with prompt length × hidden dim, typically ~100×
fewer than LoRA's low-rank adapters.

## Adaptation is first-class for applied gen AI

It would be a mistake to file PEFT under "infrastructure adjacent to gen AI."
Adaptation *is* how generative AI ships in practice:

- The Stable Diffusion ecosystem (Civitai, IP-Adapter, ControlNet, textual
  inversion) is overwhelmingly LoRA + prompt-tuning, not full-finetunes.
- Biological foundation models (scGPT, HyenaDNA, Nucleotide Transformer,
  Geneformer, Enformer-2) are unusable for most labs without PEFT — full
  fine-tuning is out of reach on commodity GPUs and the pretraining datasets
  are often unreleased.
- Domain adaptation, task adaptation, and continual learning on a frozen
  base model are the realistic deployment story for any gen-AI application
  built on top of a foundation model.

So this folder belongs in the project's gen-AI track, not adjacent to it.

## Where this sits in the adaptation stack

PEFT is one layer of a larger adaptation stack. A useful framing is three
nested levels, which **compose** rather than compete:

| Level | What it does | Examples |
|-------|--------------|----------|
| **L1 — intra-model adaptation** | Tunes *one* foundation model cheaply | LoRA, soft prompting, prefix tuning, adapters |
| **L2 — inter-model integration** | Combines multiple predictors and heterogeneous evidence into a refined prediction | Stacking, mixture-of-experts, multimodal late fusion, "meta-layer" architectures |
| **L3 — process-level orchestration** | Validates, refines, and self-improves predictions over time | Agentic workflows, retrieval-augmented validation, recursive self-improvement |

**This folder lives at L1.** Soft prompting tunes a single FM. It doesn't
fuse modalities, it doesn't combine multiple base predictors, and it
doesn't orchestrate validation agents. Those are real, valuable things —
they're just different jobs.

The sibling project **agentic-spliceai** is a worked L2+L3 example: pluggable
base predictors → 116-feature multimodal fusion → M1–M4 meta-models →
agentic validation. See its [meta_model_variants_m1_m4.md](../../../../agentic-spliceai/examples/meta_layer/docs/meta_model_variants_m1_m4.md)
for the meta-layer decomposition. Note that "meta-layer" there is
*stacking + multimodal fusion + multi-task decomposition*, not classical
MAML-style meta-learning — terminology is overloaded.

The three levels compose: agentic-spliceai's base layer can host an
FM-derived predictor that has itself been L1-adapted (e.g., LoRA-tuned
SpliceBERT), then the L2 meta-layer fuses its outputs with epigenetic /
junction / RBP evidence, and L3 agents validate the result. A serious
gen-AI biology application typically wants all three levels available.

So when this doc says soft prompting is "the right first try" for the
flagship's FM-conditioning hypothesis, that statement is scoped to L1.
The L2 question — how to fuse FM-derived sequence features with the
existing CVAE_NB / JEPA / latent-diffusion stack — is a separate design
problem that the perturbation flagship is already implicitly working on
through its three-stage architecture.

## The PEFT toolkit: soft prompting alongside LoRA

The project status table in `CLAUDE.md` lists Phase 4:

> **Phase 4 — Foundation-model adaptation framework (resource-aware configs,
> LoRA).** Status: framework done, tuning pending.

LoRA already lives under [`src/genailab/foundation/`](../../../src/genailab/foundation/).
Soft prompting is the natural complement:

| Lever | What it edits | Trainable params | Best when |
|-------|---------------|------------------|-----------|
| LoRA  | Low-rank deltas to attention/MLP weights | ~0.1–1% of base | Task needs to reshape feature distributions |
| Soft prompting | A handful of input tokens | ~0.01% of base | Task needs to *steer* a frozen model toward a known capability |
| Full fine-tune | All weights | 100% | Compute-rich, distribution shift is large |

Treating these as a toolkit — rather than picking one and committing — is
the goal. Different downstream biological tasks favour different levers.

## Steering vs reshaping — when soft prompting is the right first try

The soft-prompting-vs-LoRA tradeoff is best framed as a question about
*adaptation regime*, not which technique is "better."

**Soft prompting wins when the goal is *steering*** — drawing out a
capability the foundation model already has from pretraining:

- ~100× smaller parameter footprint than LoRA → ship thousands of
  task-specific prompts, swap them at inference
- Truly modular: prompt mixing at inference is clean, while LoRA composition
  is finicky (interpolation, ties-merging, etc.)
- Doesn't touch model internals → safe for frozen black-box FMs and works
  through input-embedding APIs
- Continual-learning friendly: task-specific prompts don't interfere with
  each other
- Pulls the model toward what it already knows rather than reshaping it

**LoRA wins when the goal is *reshaping*** — the task needs real
distribution shift away from pretraining:

- More capacity (rank-r weight delta has more room than a handful of input
  tokens)
- Better optimization stability — soft-prompt loss landscapes are notoriously
  tricky
- Gradient signal modifies internal representations at *every* adapted
  layer, not just the input
- More robust at small model scales

Mapped onto likely genai-lab adaptation needs:

| Adaptation regime | Better lever |
|-------------------|--------------|
| Steering HyenaDNA toward a regulatory-element classification task it was pretrained for | Soft prompting |
| Adapting a human-DNA FM to plant or pathogen genomes (real distribution shift) | LoRA |
| Composing many task-specific adapters on the same FM | Soft prompting |
| Perturbation-relevant sequence features from an FM (steering, not reshaping) | Soft prompting first try |

**Theoretical context.** He et al. (2022) showed that prefix tuning, LoRA,
and adapters can all be written as modifications to hidden representations
in a unified framework — they differ in *where* the modification happens.
Practically, they solve adjacent problems.

## Connection to the flagship

The perturbation-prediction flagship currently plans a CVAE_NB → JEPA →
latent diffusion stack. Soft prompting is not in that plan. It enters the
picture if and when sequence-level features from a genomic FM (HyenaDNA,
Nucleotide Transformer, Enformer-2) become useful conditioning signal for
the predictor — e.g., conditioning the perturbation embedding on the
CRISPRa-targeted gene's sequence context rather than only on gene identity.

That is a forward-looking hypothesis, not a committed milestone. Crucially,
this is exactly the *steering* regime — pulling out FM capabilities the
model already has — so soft prompting is the right first attempt, not LoRA.

## Why HyenaDNA specifically (and why it's not the point)

HyenaDNA was chosen as a demonstration substrate because:

- It is a real **genomic foundation model**, so the demo is biologically
  flavoured rather than NLP-flavoured.
- It is small enough to run on a laptop, unlike scGPT or Nucleotide
  Transformer at full size.
- It uses the Hyena operator instead of attention — a useful counter-example
  showing soft prompting doesn't depend on attention specifically; it
  depends only on the model accepting input embeddings.

The technique transfers directly to scGPT, Geneformer, Nucleotide
Transformer, and any other embedding-input foundation model. None of the
HyenaDNA-specific code in the notebook is load-bearing for the project.

## Residual caveat

Soft prompting empirically gets *better* as the base model gets bigger.
On small models (~10M–100M params, including the laptop-friendly HyenaDNA
variant) it is harder to optimize and can underperform LoRA on the same
task. So a tutorial demo on a tiny model can mislead in either direction:
weak results don't mean the technique is weak, and strong results don't
guarantee it scales linearly. Treat the laptop demo as a *concept
illustration*; defer the steering-vs-reshaping comparison to a real
downstream task on a larger FM.

## What lives here now / next

| Notebook | Status | Role |
|----------|--------|------|
| `soft_prompting_hyenadna.ipynb` | 🔬 done | Concept demo on a genomic FM |
| Soft prompting on a larger biological FM (scGPT or Nucleotide Transformer) | 🔮 future | Where the steering-vs-reshaping question gets a real answer |
| LoRA + soft-prompting comparison on the same task | 🔮 future | Toolkit-completing benchmark, ties into Phase 4 |

Status legend follows `CLAUDE.md`: 🔬 validated · 🔮 pending predecessor.

## Related

- [`src/genailab/foundation/`](../../../src/genailab/foundation/) — LoRA and
  resource-aware adaptation framework (L1); soft prompting belongs in the
  same toolkit
- [`docs/applications/perturbation_prediction.md`](../../../docs/applications/perturbation_prediction.md)
  — flagship (L2-shaped) that may eventually consume FM-derived sequence
  features
- **agentic-spliceai meta-layer** (sibling project) — worked L2+L3 example:
  base predictors + 116-feature multimodal fusion + M1–M4 meta-models +
  agentic validation. See
  [`examples/meta_layer/docs/meta_model_variants_m1_m4.md`](../../../../agentic-spliceai/examples/meta_layer/docs/meta_model_variants_m1_m4.md)
  for the meta-layer decomposition
