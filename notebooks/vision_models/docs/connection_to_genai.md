# Vision Models — Connection to genai-lab

## Why this directory exists

`notebooks/vision_models/` is the umbrella for vision-specific *generative*
architectures studied in this project. ViT is the first occupant, but it is
not the point — the point is to build up the visual-generative stack
(transformer backbones for diffusion/flow, masked autoencoders, vision
foundation models for biological imaging).

This is intentionally a category, not a single-method folder.

## Direct line to the project's generative core

The project's current image-side generative work — flow matching on MNIST —
uses a **U-Net** backbone (`VelocityUNet2D`). U-Net was the standard backbone
for DDPM and most early diffusion work. Modern image generators (SD3, FLUX,
Sora) have replaced U-Net with **transformer backbones** (DiT — Diffusion
Transformer).

The genai-lab roadmap reflects this. From the project status table in
`CLAUDE.md`:

> Research prototypes with theory complete, implementation pending: **DiT,
> Flow Matching / Rectified Flow, Latent Diffusion with NB/ZINB decoders,
> EBMs.**

`mini_vit_from_scratch.ipynb` builds the exact pieces that DiT is made of:

| ViT component | Re-used in DiT | Re-used in flow-matching with transformer backbone |
|---------------|----------------|----------------------------------------------------|
| Patch embedding (Conv2d stride=patch_size) | ✅ tokenizes the noisy image | ✅ same |
| Class token / positional embedding | ✅ + adaLN-Zero conditioning | ✅ + time conditioning |
| Transformer block (MSA + MLP + LayerNorm) | ✅ with adaLN-Zero | ✅ same |
| Final classifier head | ❌ replaced by a velocity/noise prediction head | ❌ same |

So the ViT notebook is **prerequisite scaffolding** for the DiT-on-MNIST
follow-up, which is the natural next notebook in this directory. That
follow-up will also enable a U-Net vs Transformer head-to-head comparison on
the same flow-matching objective (same loss, same data, same sampler — only
the velocity network changes).

## Why "vision" matters for computational biology

Generative AI for biology is not only sequence modelling. Two areas where
vision-style generative models are directly load-bearing:

1. **Spatial transcriptomics.** Visium, Slide-seq, MERFISH, Stereo-seq all
   produce data on a 2-D spatial grid — gene expression with image-like
   structure. Patch-based encoders are the natural fit; ViT-style tokenization
   carries over directly.
2. **Microscopy / cell painting.** Phenotypic screens (Cell Painting JUMP-CP,
   the IDR repository) yield large image collections. Generative and
   self-supervised vision models (DINOv2, MAE, diffusion priors) are the
   current best practice for representation learning here, and several biology
   foundation models (Phenom, MicroSAM, BioCLIP) are vision-first.

These are not on the immediate roadmap, but they are why "vision" deserves a
dedicated category rather than being absorbed into `notebooks/diffusion/` or
similar.

## What lives here now / next

| Notebook | Status | Role |
|----------|--------|------|
| `mini_vit_from_scratch.ipynb` | 🔬 done | ViT internals on MNIST (classifier — *discriminative*, not generative) |
| `dit_on_mnist.ipynb` | 🎯 planned next | DiT replaces U-Net in flow-matching; same loss, transformer velocity field |
| MAE / masked image modelling | 📝 future | Self-supervised pretraining for downstream tasks |
| Vision FM adaptation (cell imaging) | 🔮 future | If/when spatial or microscopy work joins the flagship track |

Status legend follows `CLAUDE.md`: 🔬 validated · 🎯 active · 📝 prototype
planned · 🔮 pending predecessor.

## Honest caveat

The current ViT notebook is a **classifier**, not a generative model. Its
value to genai-lab is via the *building blocks* it makes concrete, not the
end-to-end model. The folder earns its keep when the DiT notebook lands and
closes the loop from "ViT internals" to "ViT generating images."

## Related

- [`docs/applications/perturbation_prediction.md`](../../../docs/applications/perturbation_prediction.md)
  — flagship that consumes the generative stack
- [`src/genailab/flow_matching/velocity_networks.py`](../../../src/genailab/flow_matching/velocity_networks.py)
  — current U-Net velocity field; DiT will live alongside it
- [`dev/flow_matching/QA/02_flow_matching_mnist_warmup/velocity_unet2d_architecture.md`](../../../dev/flow_matching/QA/02_flow_matching_mnist_warmup/velocity_unet2d_architecture.md)
  — companion U-Net walk-through
