# Soft Prompting — Connection to genai-lab

## Why this directory exists

`notebooks/soft_prompting/` covers **soft prompting** as a parameter-efficient
fine-tuning (PEFT) technique. The current notebook uses HyenaDNA as a
substrate, but the technique — not the specific foundation model — is the
subject.

Soft prompting prepends a small number of *learned* token embeddings to a
frozen foundation model's input. Only the prompt embeddings are trained.
Trainable parameters scale with prompt length × hidden dim, typically far
fewer than LoRA's low-rank adapters.

## Direct line to the project's foundation-model adaptation track

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

Treating these as a toolkit (rather than picking one and committing) is the
goal — different downstream biological tasks will favour different levers.

## Connection to the flagship

The perturbation-prediction flagship currently plans a CVAE_NB → JEPA → latent
diffusion stack. Soft prompting is not in that plan. It enters the picture if
and when sequence-level features from a genomic foundation model (HyenaDNA,
Nucleotide Transformer, Enformer-2) become useful conditioning signal for the
perturbation predictor — e.g., conditioning a CVAE's perturbation embedding
on the CRISPRa-targeted gene's sequence context rather than only the gene
identity.

That is a forward-looking hypothesis, not a committed milestone. The
soft-prompting notebook earns its keep by making the technique concrete now,
so it is available when the flagship needs it.

## Why HyenaDNA specifically (and why it's not the point)

HyenaDNA was chosen as a demonstration substrate because:

- It is a real **genomic foundation model**, so the demo is biologically
  flavoured rather than NLP-flavoured.
- It is small enough to run on a laptop, unlike scGPT or Nucleotide
  Transformer at full size.
- It uses the Hyena operator instead of attention — a useful counter-example
  that shows soft prompting doesn't depend on attention specifically; it
  depends on the model accepting input embeddings.

The technique transfers directly to scGPT, Geneformer, Nucleotide
Transformer, and any other embedding-input foundation model. None of the
HyenaDNA-specific code in the notebook is load-bearing for the project.

## Honest caveat: where this connects to "gen AI"

Soft prompting itself is **not a generative technique**. It is a PEFT method.
Its connection to genai-lab is indirect:

- Foundation models, including the genomic ones above, are commonly trained
  with autoregressive or masked generative objectives. The pretraining that
  makes them useful is generative; soft prompting adapts them.
- HyenaDNA in particular is autoregressive — generative in the LLM sense, not
  in the diffusion/VAE/flow sense that dominates the rest of this project.

So this folder is best understood as part of the **adaptation toolkit** for
generative foundation models, not as a generative-modelling track in its own
right. If a future notebook applies soft prompting to a genuinely generative
biological model (e.g., a sequence-conditioned CVAE_NB or a diffusion-based
perturbation predictor), that will be the cleaner generative-AI connection.

## What lives here now / next

| Notebook | Status | Role |
|----------|--------|------|
| `soft_prompting_hyenadna.ipynb` | 🔬 done | Concept demo on a genomic FM |
| Soft prompting on a generative biological model | 🔮 future | Closes the loop with the project's main thrust |
| LoRA + soft-prompting comparison on the same task | 🔮 future | Toolkit-completing benchmark, ties into Phase 4 |

Status legend follows `CLAUDE.md`: 🔬 validated · 🔮 pending predecessor.

## Related

- [`src/genailab/foundation/`](../../../src/genailab/foundation/) — LoRA and
  resource-aware adaptation framework; soft prompting belongs in the same
  toolkit
- [`docs/applications/perturbation_prediction.md`](../../../docs/applications/perturbation_prediction.md)
  — flagship that may eventually consume FM-derived sequence features
