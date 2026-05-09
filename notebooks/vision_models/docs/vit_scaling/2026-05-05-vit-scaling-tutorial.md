# Scaling Vision Transformers: From a Toy ViT to Production-Grade Vision Systems

You can understand the whole scaling problem of Vision Transformers from one innocent line in the MiniViT notebook:

```python
attn = (q @ k.transpose(-2, -1)) * scale
```

That line is beautiful. It is also where vanilla ViT starts to break.

In the notebook, we train a small ViT on CIFAR-10. The image is only `32×32`, the patch size is `4×4`, and the model sees only:

```text
32 / 4 = 8 patches per side
8 × 8 = 64 image tokens
+ 1 CLS token = 65 total tokens
```

An attention matrix of `65 × 65` is tiny. You can print it, visualize it, and reason about it directly. That is exactly why the notebook is a good learning artifact.

But production vision systems do not usually live at CIFAR scale. Medical imaging, satellite imagery, pathology, document AI, high-resolution detection, and segmentation all push the token count much higher. Once the token count grows, the same elegant attention line becomes expensive because self-attention scales as:

```text
O(N²)
```

where `N` is the number of tokens.

This tutorial connects the clean MiniViT implementation to the scaling tricks used in real systems.

---

## 1. The MiniViT mental model

The notebook builds ViT from first principles:

```text
Image
  → patch embedding
  → CLS token + positional embedding
  → transformer encoder blocks
  → classification head
```

The key move is to turn an image into a sequence.

For an image of size `H × W` and patch size `P`, the number of patch tokens is:

```text
N = (H / P) × (W / P)
```

For CIFAR-10:

```text
H = W = 32
P = 4
N = (32 / 4)² = 64
```

The notebook uses a useful PyTorch idiom:

```python
nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

This is equivalent to extracting non-overlapping patches, flattening each patch, and applying a linear projection. The convolution is not being used as a classic CNN feature extractor here. It is being used as a fast patch tokenizer.

That is the first bridge from CNNs to ViTs:

```text
CNN view: local filters over pixels
ViT view: patch tokenizer that creates a sequence
```

---

## 2. Where vanilla ViT breaks

Now scale the image.

For a `1024×1024` image with a ViT-style patch size of `16`:

```text
N = (1024 / 16)²
  = 64²
  = 4096 tokens
```

The attention matrix is:

```text
4096 × 4096 = 16,777,216 entries
```

That is per head, per layer, per sample.

If stored as `float32`:

```text
16,777,216 × 4 bytes ≈ 67 MB
```

Now multiply that by several heads, several layers, batch size, gradients, optimizer state, and intermediate activations. The toy notebook becomes a memory problem very quickly.

The important point is not merely “large images are large.” The deeper point is:

> ViT does not scale with image area directly. It scales with the square of the number of patch tokens.

If you double image height and width, the number of tokens grows by `4×`, but the attention matrix grows by `16×`.

That is the production pain.

---

## 3. The three knobs you can turn

Production ViT systems usually scale by changing one or more of these:

### Knob 1: Reduce the number of tokens

Use larger patches, downsampling, pooling, or a CNN stem.

Example:

```text
1024×1024 image, patch size 16 → 4096 tokens
1024×1024 image, patch size 32 → 1024 tokens
```

That is a `4×` reduction in tokens and a `16×` reduction in attention matrix size.

The cost: larger patches lose fine-grained spatial detail.

This is dangerous in domains like pathology or radiology, where tiny structures can matter.

### Knob 2: Restrict who attends to whom

Instead of letting every token attend to every other token, use local windows.

This is the Swin Transformer idea:

```text
global attention over all patches
→ local attention inside windows
→ shifted windows to communicate across boundaries
```

The model keeps the inductive bias that nearby pixels are often related, while still allowing information to move across the image over depth.

### Knob 3: Approximate or replace attention

Instead of computing the full `N × N` attention matrix, use a cheaper approximation or a different token-mixing mechanism.

Examples:

- Linformer: project keys and values into a lower-rank sequence dimension.
- Performer: approximate softmax attention using random feature maps.
- FNet: replace attention with Fourier mixing.

These are especially important as conceptual tools: they show that the bottleneck is not “transformers” in general, but dense pairwise token interaction.

### Research-backed map of the design space

After checking the primary sources, the four methods separate cleanly into different families:

| Method | What changes? | Core scaling move | Best mental model | Main caveat |
|---|---|---|---|---|
| **Swin Transformer** | Attention pattern + visual hierarchy | Local window attention with shifted windows; patch merging between stages | CNN-like feature pyramid built from transformer blocks | Strong for vision backbones, but no longer “pure flat ViT” |
| **Linformer** | Attention approximation | Project keys/values along sequence length; exploit low-rank structure | Attend to a compressed memory of the sequence | Compression can lose fine pairwise detail |
| **Performer** | Attention algebra | Approximate softmax attention with FAVOR+ random features | Keep global attention, avoid materializing the all-pairs matrix | Approximation quality and numerical stability matter |
| **FNet** | Replace attention mixer | Use fixed Fourier transform token mixing | Attention is one mixer, not the only mixer | Less adaptive than learned attention |

The important distinction: Swin is not merely an “efficient attention trick.” It is a vision-backbone redesign. Linformer and Performer are closer to attention substitutions. FNet is a token-mixing replacement.

---

## 4. Strategy: larger patches

The simplest way to reduce memory is to increase patch size.

```text
Image: 1024×1024
Patch size 16 → 4096 tokens
Patch size 32 → 1024 tokens
Patch size 64 → 256 tokens
```

This is easy to implement:

```python
patch_embed = nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=32,
    stride=32,
)
```

But larger patches are a blunt instrument.

A `32×32` patch may be fine for coarse classification, but not for dense segmentation, lesion detection, pathology nuclei, tiny fractures, or small object recognition. You are asking one token to summarize a larger piece of the image.

Good use case:

```text
Image-level classification where global semantics dominate.
```

Risky use case:

```text
Fine-grained localization or medical diagnosis where small morphology matters.
```

---

## 5. Strategy: hierarchical vision transformers

The original ViT treats the image as one flat sequence at one resolution.

That is unlike CNNs. CNNs naturally form a pyramid:

```text
early layers: high resolution, local features
middle layers: lower resolution, larger receptive field
late layers: semantic/global features
```

Hierarchical ViTs borrow this pyramid idea.

Instead of keeping the same token grid throughout the network, they progressively reduce spatial resolution and increase channel depth.

A rough picture:

```text
Stage 1: many tokens, small channels, local detail
Stage 2: fewer tokens, more channels
Stage 3: even fewer tokens, more semantics
Stage 4: compact global representation
```

Swin Transformer is the canonical example.

Its key ideas:

1. Split the image into non-overlapping windows.
2. Apply self-attention only inside each window.
3. Shift the windows in alternating layers.
4. Merge patches between stages to reduce resolution.

The shift is the clever part. If windows never moved, tokens near different window boundaries would not communicate well. Shifted windows create cross-window connections without paying full global attention cost.

You can think of it like neighborhood gossip:

```text
Layer 1: each neighborhood talks internally.
Layer 2: the neighborhood boundaries shift, so new people meet.
Layer 3+: information gradually spreads across the whole city.
```

This is why Swin became so influential for detection, segmentation, and high-resolution medical imaging. It behaves more like a CNN pyramid while preserving transformer-style token mixing.

The original Swin paper makes three claims worth remembering:

- It is designed as a **general-purpose vision backbone**, not just an image classifier.
- Shifted windows give cross-window communication while keeping attention local.
- Because the model is hierarchical, it naturally supports dense prediction tasks such as detection and segmentation.

That last point is the production lesson. Classification can sometimes tolerate a flat global token sequence. Segmentation and detection usually want multi-scale feature maps. Swin gives transformers that CNN-like multi-scale shape.

---

## 6. Strategy: CNN stem → ViT

Another production pattern is to use a small CNN at the front before the transformer.

Instead of immediately converting raw pixels into ViT tokens, the model first downsamples and extracts local features:

```text
image
  → conv stem
  → smaller feature map
  → patch/token embedding
  → transformer
```

This helps because CNNs are efficient at local processing. They also inject useful visual inductive biases:

- locality
- translation equivariance
- edge/texture sensitivity
- progressive downsampling

A simplified example:

```python
stem = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.GELU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.GELU(),
)
```

A `1024×1024` image becomes a `256×256` feature map after two stride-2 convolutions. The transformer now sees a much smaller spatial grid.

This is the hybrid philosophy:

> Let convolution handle cheap local visual processing; let attention handle flexible long-range interaction.

Architectures like CvT, CoAtNet, and EfficientViT live in this design space.

---

## 7. Strategy: tiling and aggregation

Some images are too large to treat as a single tensor.

Whole-slide pathology images can be around `100,000×100,000` pixels. There is no practical vanilla ViT sequence for that.

The standard pipeline is more like this:

```text
whole-slide image
  → tile into many 256×256 or 512×512 crops
  → encode each tile with a pretrained vision model
  → aggregate tile embeddings
  → slide-level prediction
```

The tile encoder might be a ViT, ConvNeXt, DINO-style model, UNI, CONCH, PathDINO, or another pathology foundation model.

The aggregator might be:

- attention-based multiple-instance learning
- sparse transformer
- graph model
- Mamba/state-space model
- clustering + pooling

This is a different meaning of “patch.”

In vanilla ViT, a patch is a small region inside one image tensor.

In pathology, a tile is often an image-level crop from a massive slide. Each tile gets its own embedding, and the slide is modeled as a bag or sequence of tile embeddings.

This distinction matters.

```text
ViT patch: token inside an image
WSI tile: image crop inside a slide
```

---

## 8. Strategy: efficient attention variants

Efficient attention methods ask a different question:

> Can we preserve most of what attention gives us without explicitly building the full N×N matrix?

Three useful reference points are Linformer, Performer, and FNet.

### Linformer: attention is approximately low-rank

Linformer assumes the full attention pattern has redundancy. Instead of attending over all `N` keys and values directly, it projects the sequence length down to a smaller dimension `k`:

```text
K: N × d  →  k × d
V: N × d  →  k × d
```

Then attention becomes roughly:

```text
Q K_compressedᵀ
```

So instead of an `N × N` attention matrix, you get something closer to `N × k`, where `k << N`.

Mental model:

```text
Do not ask every token to compare against every original token.
Ask it to compare against a compressed summary of the token sequence.
```

### Performer: approximate softmax attention with random features

Performer uses a kernel trick. Softmax attention can be viewed as a kernel operation. Performer approximates that kernel using random feature maps, often summarized as FAVOR+.

Instead of explicitly forming:

```text
softmax(QKᵀ)V
```

it rewrites the computation so the expensive pairwise matrix is avoided.

Mental model:

```text
Map Q and K into a feature space where attention can be computed by associativity.
```

The advantage is linear complexity in sequence length under the approximation.

The tradeoff is that approximation quality and numerical stability matter.

The paper-level distinction from Linformer is subtle but important:

- Linformer assumes attention can be approximated as **low-rank**.
- Performer does **not** rely on low-rankness or sparsity; it approximates the attention kernel using positive orthogonal random features.

So Performer is not “compress the sequence first.” It is “rewrite the attention computation so the full pairwise matrix never has to appear.”

### FNet: maybe attention is not always necessary

FNet is more radical. It replaces self-attention with Fourier transforms for token mixing.

Instead of learning pairwise attention weights, it applies Fourier mixing across the sequence and hidden dimensions.

Mental model:

```text
Attention learns how tokens mix.
Fourier mixing uses a fixed global mixing operation.
```

This can be fast and surprisingly competitive in some language settings, but it is less expressive than learned attention. For vision, it is best understood as part of the broader family of token-mixing alternatives.

The FNet result is useful because it weakens an assumption people often make too quickly: maybe the benefit of transformer encoders is not only “learned attention,” but also repeated global token mixing plus nonlinear feed-forward processing. FNet reportedly reached 92–97% of BERT accuracy on GLUE while training much faster in the original study. That does not make it a universal replacement for attention, but it is a sharp conceptual counterexample.

---

## 9. How this connects back to the notebook

The MiniViT notebook is the clean base case.

It teaches the invariant pieces:

- patch tokenization
- sequence length
- positional information
- Q/K/V attention
- multi-head reshape logic
- residual transformer blocks
- CLS-based classification

Scaling methods mostly modify only two parts of that story:

```text
1. How tokens are created.
2. How tokens communicate.
```

Everything else is a variation.

Examples:

```text
Vanilla ViT:
  tokens = fixed non-overlapping patches
  communication = global attention

Swin:
  tokens = hierarchical patch stages
  communication = local window attention + shifted windows

CNN stem + ViT:
  tokens = features from downsampled conv maps
  communication = transformer attention after local preprocessing

Linformer:
  tokens = usually same as transformer input
  communication = attention against compressed keys/values

Performer:
  tokens = usually same as transformer input
  communication = approximate kernelized attention

FNet:
  tokens = same sequence abstraction
  communication = Fourier token mixing instead of attention
```

That is the conceptual compression I would keep in your head.

---

## 10. Practical decision guide

If the image is small and classification is the goal:

```text
Vanilla ViT is fine for learning and experimentation.
```

If the image is high-resolution but the label is global:

```text
Try larger patches, CNN stem, or hierarchical ViT.
```

If spatial localization matters:

```text
Prefer hierarchical models like Swin, or encoder-decoder variants such as SwinUNETR.
```

If the image is enormous, like pathology WSI:

```text
Use tiling + tile encoder + slide-level aggregation.
```

If the sequence is long because of many tokens and you want to research alternatives:

```text
Study Linformer, Performer, FNet, sparse attention, Mamba, and other token-mixing variants.
```

My take: for production vision, Swin-style hierarchy and CNN/ViT hybrids are often more practical than pure efficient-attention theory. Linformer/Performer/FNet are conceptually important because they teach you how people attack the `N²` bottleneck, but high-resolution vision systems usually need architectural hierarchy, not only cheaper attention.

A useful practical hierarchy:

1. **Need a strong vision backbone for detection/segmentation?** Start with Swin-style hierarchy or another hierarchical ViT.
2. **Need long-sequence transformer research intuition?** Study Linformer and Performer.
3. **Need to understand attention alternatives broadly?** Study FNet, MLP-Mixer, state-space models, and convolutional token mixers.
4. **Need WSI/pathology scale?** Do not force a monolithic ViT. Use tiling, pretrained tile encoders, and slide-level aggregation.

---

## 11. Interview-ready explanation

A compact way to explain ViT scaling:

> A vanilla ViT turns an image into a flat sequence of patch tokens and applies global self-attention. This is elegant, but attention scales quadratically with the number of tokens. For high-resolution images, the token count grows quickly, so the attention matrix becomes too expensive. Production systems usually solve this by reducing tokens with larger patches or CNN stems, restricting attention with local windows as in Swin Transformer, building hierarchical feature pyramids, or avoiding full attention through approximations like Linformer and Performer. In pathology and other extreme-resolution domains, the standard solution is often tiling plus aggregation rather than applying one giant ViT to the whole image.

---

## Source context

This note was built from:

- `/Users/pleiadian53/work/genai-lab/notebooks/vision_models/mini_vit_from_scratch.ipynb`
- `/Users/pleiadian53/work/genai-lab/dev/planning/flow_matching/memo/ViT-scaling.md`

Primary concepts referenced and web-checked:

- Vision Transformer / ViT: Dosovitskiy et al., “An Image is Worth 16x16 Words” — https://arxiv.org/abs/2010.11929
- Swin Transformer: Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows” — https://arxiv.org/abs/2103.14030
- Linformer: Wang et al., “Linformer: Self-Attention with Linear Complexity” — https://arxiv.org/abs/2006.04768
- Performer: Choromanski et al., “Rethinking Attention with Performers” — https://arxiv.org/abs/2009.14794
- FNet: Lee-Thorp et al., “FNet: Mixing Tokens with Fourier Transforms” — https://research.google/pubs/fnet-mixing-tokens-with-fourier-transforms/
