# Swin Transformer

## One-sentence version

Swin Transformer is a hierarchical Vision Transformer that makes high-resolution vision practical by doing self-attention inside local windows, then shifting those windows across layers so information can move between regions.

## Why it exists

Vanilla ViT uses global self-attention:

```text
every patch attends to every other patch
```

That is elegant, but expensive:

```text
attention cost ≈ O(N²)
```

For high-resolution images, `N` gets large fast. Swin asks: what if we keep attention, but make it local and hierarchical like a CNN feature pyramid?

## Core mechanism

Swin has three main ideas:

### 1. Window attention

Instead of computing attention over the full image token grid, split the image into windows.

```text
full token grid → many small windows
```

Each token attends only to tokens inside its window.

If a window has `M × M` tokens, the attention matrix is only:

```text
M² × M²
```

not:

```text
N × N
```

### 2. Shifted windows

If windows stayed fixed forever, information would be trapped inside each window. Swin solves this by shifting the window partition in alternating layers.

```text
Layer 1: regular windows
Layer 2: shifted windows
Layer 3: regular windows
Layer 4: shifted windows
```

This lets neighboring windows exchange information without full global attention.

### 3. Patch merging

Between stages, Swin merges neighboring patches, reducing spatial resolution and increasing channel depth.

This creates a pyramid:

```text
many small tokens → fewer richer tokens → fewer more semantic tokens
```

That is why Swin works well as a backbone for detection and segmentation, not just classification.

## Intuition

Swin is like a city gossip network:

- In one layer, each neighborhood talks internally.
- In the next layer, the neighborhood boundaries shift.
- Over many layers, information travels across the city.

You never need one giant meeting where every person talks to every other person.

## How it connects to vanilla ViT

Vanilla ViT:

```text
flat sequence + global attention
```

Swin:

```text
hierarchical grid + local window attention + shifted boundaries
```

The transformer block is still recognizable: attention, MLP, residuals, layer norm. The big change is the attention pattern and the multi-stage visual hierarchy.

## Why it matters for medical imaging

Medical images are often large, and small structures matter. Simply increasing patch size may erase useful detail.

Swin-style hierarchy is attractive because it keeps local detail early while still building larger context later.

This is why models like SwinUNETR became important for medical segmentation.

## Ripple and connect

Related ideas:

- **CNN feature pyramid**: Swin borrows the idea of progressively lower-resolution, higher-level features.
- **Local attention**: reduce attention cost by restricting neighborhoods.
- **Shifted windows**: boundary-crossing trick that avoids isolated local windows.
- **U-Net**: encoder-decoder segmentation architecture; Swin can act as a transformer encoder inside U-Net-like systems.
- **Inductive bias**: Swin reintroduces locality and hierarchy, which vanilla ViT mostly lacks.

Useful idioms/phrases:

- “Swin makes attention local without making the model blind to global context.”
- “Shifted windows are the communication bridge between local regions.”
- “Swin is closer to a CNN pyramid than to a flat BERT-style ViT.”

## Interview-ready explanation

Swin Transformer scales ViT to high-resolution vision by replacing global self-attention with window-based local attention. To avoid isolating each window, it shifts the window partition in alternating layers, allowing cross-window communication. It also uses patch merging to form a hierarchical feature pyramid, making it useful for dense prediction tasks like detection and medical segmentation.

## Research notes from the Swin paper

The Swin paper frames the model as a **general-purpose backbone for computer vision**. That matters because the goal is not merely image classification; the architecture is designed to produce multi-scale features useful for detection and segmentation.

Key claims from the abstract:

- Shifted windows limit self-attention to local non-overlapping windows while enabling cross-window connection.
- The architecture has linear computational complexity with respect to image size.
- The hierarchical design makes it compatible with broad vision tasks: classification, object detection, instance segmentation, and semantic segmentation.

### Complexity intuition

If there are `N` total tokens, global attention creates an `N × N` matrix. Swin instead splits tokens into windows. If each window has `M²` tokens, attention is computed inside each window. The total cost grows roughly with the number of windows times the per-window attention cost, which is much gentler than full global attention when `M` is fixed.

### What to remember

Swin is a production-style answer to high-resolution vision because it changes the **geometry** of the transformer. It does not merely approximate attention; it gives the model a CNN-like pyramid and locality bias.

## Source

- Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows” — https://arxiv.org/abs/2103.14030
