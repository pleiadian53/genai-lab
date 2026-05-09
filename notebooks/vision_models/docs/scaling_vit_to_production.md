# Scaling ViT to Production: Strategies and Domain Guidance

*Companion to `mini_vit_from_scratch.ipynb`*

---

## Why Vanilla ViT Breaks at High Resolution

The patch embedding layer (`Conv2d(kernel_size=P, stride=P)`) is efficient at any resolution.
The true bottleneck is **self-attention**, which is \(O(N^2)\) in time and memory, where
\(N\) is the number of patch tokens:

$$N = \frac{H}{P} \times \frac{W}{P}$$

| Image size | Patch size | Tokens \(N\) | Attention matrix size (float32) |
|---|---|---|---|
| 32 × 32 | 4 | 64 | 16 KB |
| 224 × 224 | 16 | 196 | 154 KB |
| 512 × 512 | 16 | 1 024 | 4 MB |
| 1024 × 1024 | 16 | 4 096 | 64 MB |
| 4096 × 4096 | 16 | 65 536 | 16 GB |

At 1024 px with ViT-L (16 heads × 24 layers) the memory for attention maps alone is
\(16 \times 24 \times 64\text{ MB} \approx 24\text{ GB}\), which exceeds a typical A100.
Every strategy below attacks this \(O(N^2)\) problem differently.

---

## Strategy Catalogue

### 1. Larger Patch Size

**How it works.** Simply increase \(P\). A 1024 px image with \(P=32\) produces only 1024 tokens
instead of 4096. The Conv2d embedding is unchanged; only the hyperparameter changes.

**Trade-offs.**

- ✅ Zero architecture change — drop-in for any ViT codebase.
- ✅ Quadratic savings are large: doubling \(P\) cuts \(N\) by 4×.
- ❌ Each patch covers more pixels, so the model loses fine-grained spatial detail.
- ❌ Poor for tasks that require localisation (detection, segmentation, dense prediction).

**When to use.** Image-level classification where global semantics dominate and fine detail is
secondary. A fast baseline before investing in more complex architectures.

---

### 2. Hierarchical Windowed Attention — Swin Transformer

**How it works.** The Swin Transformer (Liu et al., 2021) replaces global self-attention with
*local window attention*: each token only attends to the \(M \times M\) window it falls in
(typically \(M=7\)). Cross-window communication is achieved by *shifting* the window grid by
\((M/2, M/2)\) every other layer. Feature maps are progressively downsampled (like a CNN
pyramid), so early layers process fine features and late layers process coarse semantic features.

```
Stage 1: patch size=4, resolution H/4 × W/4, window size 7×7
Stage 2: merge 2×2 patches → H/8 × W/8
Stage 3: merge 2×2 patches → H/16 × W/16
Stage 4: merge 2×2 patches → H/32 × W/32
```

Attention complexity is \(O(N)\) because the window size is fixed regardless of image size.

**Trade-offs.**

- ✅ Linear complexity — handles 1024 px and beyond.
- ✅ Multi-scale features — ideal for dense prediction tasks.
- ✅ Best single model on COCO detection/segmentation benchmarks for years.
- ❌ More complex to implement than plain ViT (window partitioning, shift masking).
- ❌ Window locality can miss very long-range dependencies.

**Key variants.** Swin-T/S/B/L, Swin-V2, Swin3D (video/3D volumes).

**When to use.** The default choice for object detection, instance segmentation, and semantic
segmentation in standard computer vision. Backbone in DINO-v2, Mask DCHM, and many SOTA models.

---

### 3. CNN Stem → ViT Hybrid

**How it works.** Replace the single Conv2d patch embedding with a small CNN (typically 3–5
conv layers) that aggressively downsamples the image before the transformer sees it. The
transformer then operates on a much smaller spatial feature map.

```
Input (B, 3, H, W)
   → CNN stem: 3–5 conv layers with stride 2 each → (B, C, H/32, W/32)
   → Flatten spatial dims → (B, N_small, C)
   → Transformer encoder
```

For a 1024 px image with a ×32 stem: \(N = 32 \times 32 = 1024\) tokens — manageable.

The CNN provides inductive biases (local connectivity, translation equivariance) that benefit
the early layers, while the transformer handles global reasoning on the compressed representation.

**Notable models.** `CvT` (Conv-in-Transformer), `CoAtNet`, `EfficientViT`, `CMT`,
`LeViT`. Apple's `MobileViT` uses interleaved conv and transformer blocks.

**Trade-offs.**

- ✅ Strong performance, especially with limited data (CNN inductive bias helps).
- ✅ Very practical: the transformer processes a small map, so inference is fast.
- ✅ Easy to swap in a pretrained CNN stem.
- ❌ Loses the "pure attention" property; harder to analyse theoretically.
- ❌ Two components to tune (CNN hyperparameters + transformer hyperparameters).

**When to use.** Mobile/edge deployment, limited training data, or when you want ViT-class
accuracy with fewer FLOPs. Common in production vision APIs.

---

### 4. Linear / Efficient Attention

**How it works.** Approximate the softmax attention kernel \(\text{softmax}(QK^T/\sqrt{d})V\)
with a kernel decomposition:

$$\text{Attention}(Q, K, V) \approx \phi(Q) \left(\phi(K)^T V\right)$$

where \(\phi\) is a low-rank feature map. The key insight: \(\phi(K)^T V\) can be computed
once (\(O(Nd)\)), and then \(\phi(Q)(\phi(K)^T V)\) is a matrix–vector product per query —
reducing overall complexity from \(O(N^2 d)\) to \(O(Nd^2)\).

**Notable models.** `Linformer` (low-rank projection of K/V),
`Performer` (random Fourier features approximation),
`FNet` (replace attention with FFT entirely — not an approximation but a departure).

**Trade-offs.**

- ✅ Linear complexity.
- ❌ Approximation introduces a quality gap vs exact attention.
- ❌ Less widely adopted for vision than for NLP (Swin fills this niche better in CV).
- ❌ Random feature methods can be numerically unstable.

**When to use.** Very long sequences where even Swin is too expensive. Research context;
rarely the first choice in applied vision today.

---

### 5. Flash Attention (Implementation-Level Optimization)

**How it works.** Flash Attention (Dao et al., 2022; v2 in 2023; v3 in 2024) does NOT reduce
the algorithmic complexity — it is still \(O(N^2)\). Instead it reorders computation to be
**IO-aware**: it tiles the attention matrix to fit in SRAM (on-chip cache), avoiding repeated
reads/writes to slow HBM (GPU memory). The result is 2–4× speedup and up to 10× memory
reduction in practice.

```python
# In PyTorch ≥ 2.0, scaled_dot_product_attention automatically uses Flash Attention
# when inputs are on CUDA and no attention mask is provided.
out = F.scaled_dot_product_attention(q, k, v)  # uses Flash Attention kernel on CUDA
```

**Trade-offs.**

- ✅ Drop-in replacement — no architecture change, no quality loss.
- ✅ Enables larger batch sizes and longer sequences on the same hardware.
- ❌ Does not change the \(O(N^2)\) limit — you still cannot process WSI-scale sequences.
- ❌ Full benefit only on CUDA (NVIDIA); MPS support is partial.

**When to use.** Always, as a baseline optimisation. Replace your attention implementation
with `F.scaled_dot_product_attention` as a first step before any architectural change.

---

### 6. Tiling + Aggregation Pipeline (for very high resolution)

This is the dominant strategy when images are so large that even Swin cannot process them
end-to-end. Details in the **Medical Imaging** section below, but the pattern applies broadly
to satellite imagery, aerial photography, and industrial inspection.

---

## Recommendation by Domain

### Standard Computer Vision (ImageNet-scale, detection, segmentation)

| Task | Recommended approach | Notes |
|---|---|---|
| Image classification | Plain ViT + Flash Attention, or DINOv2 fine-tune | 224–384 px; sequence length manageable |
| Object detection | Swin Transformer backbone + DINO/DETA head | Linear complexity; SOTA on COCO |
| Semantic segmentation | Swin + UperNet, or SegFormer | Multi-scale features critical |
| Instance segmentation | Swin + Mask R-CNN or Mask2Former | |
| Mobile / edge | MobileViT, EfficientViT, CoAtNet-0 | CNN stem hybrid |
| Video understanding | Swin-3D or VideoMAE (temporal tubes) | |

**Practical starting point:** Fine-tune a DINOv2 (ViT-B/14) backbone using
`F.scaled_dot_product_attention`. DINOv2 features transfer extremely well with minimal
fine-tuning data. If you need dense prediction, switch to a Swin backbone.

---

### Medical Imaging

Medical images span a wide range. The right strategy depends on the modality:

| Modality | Typical size | Recommended approach |
|---|---|---|
| Radiology (X-ray, CXR) | 512–2048 px, 2D | Swin or ViT-B with moderate patch size |
| MRI / CT slices | 256–512 px, 2D or 3D | SwinUNETR (3D), nnFormer |
| Dermatology | 512–1024 px | CNN stem hybrid or Swin-S |
| Pathology (WSI) | 50,000–100,000 px | **Tiling + MIL pipeline** (see below) |
| Retinal fundus | 1024–4096 px | Swin or tiling for ultra-high res |

For 3D volumes (CT, MRI), use **3D patch embedding** (`Conv3d(kernel_size=(P,P,P), stride=(P,P,P))`)
— the same Conv trick extended to three dimensions. SwinUNETR and nnFormer are the
go-to architectures for volumetric segmentation.

---

## Deep Dive: The WSI Tiling Pipeline for Computational Pathology

Whole slide images (WSIs) in digital pathology are scanned at 20× or 40× magnification,
producing images of 50,000–100,000 px per side — roughly 5–10 gigapixels per slide.
End-to-end training on a full slide is physically impossible. The field has converged on a
**three-stage pipeline**.

```
┌──────────────────────────────────────┐
│  Stage 1 — Tile Extraction           │
│                                      │
│  WSI (100,000 × 100,000 px)          │
│    ↓  tissue segmentation            │
│    ↓  grid tiling (256 or 512 px)    │
│  N tiles  (N ≈ 1,000–50,000)         │
└──────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  Stage 2 — Tile Encoding             │
│                                      │
│  Each tile: (3, 256, 256)            │
│    → pre-trained encoder             │
│  Each tile: feature vector (1, D)    │
│  D = 768 (ViT-B) or 1024 (ViT-L)    │
│                                      │
│  This stage is run ONCE and cached.  │
└──────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  Stage 3 — Slide-Level Aggregation   │
│                                      │
│  Bag of N embeddings: (N, D)         │
│    → MIL pooling / sparse attn       │
│  Slide-level prediction              │
└──────────────────────────────────────┘
```

### Stage 1 — Tile Extraction in Detail

Before tiling, **tissue segmentation** removes background (glass, pen marks, white space).
Otsu thresholding on the thumbnail level (a low-resolution overview pyramid layer) is
standard. Only tiles that fall on tissue are extracted, which reduces \(N\) dramatically.

```python
# Conceptual pseudocode (real code uses openslide-python or tiatoolbox)
import openslide

wsi = openslide.OpenSlide("slide.svs")
thumbnail = wsi.get_thumbnail((512, 512))            # fast overview
tissue_mask = otsu_segment(thumbnail)                # binary mask
tile_coords = grid_coords(wsi, tile_size=256, level=1)  # 20× magnification
tiles = [wsi.read_region(xy, level=1, size=(256,256))
         for xy in tile_coords if tissue_mask[xy]]   # skip background
```

Tile size of **256 px** at 20× corresponds to roughly **0.5 μm/pixel × 256 = 128 μm**
of tissue — captures a few dozen cells, which is the right scale for histology patterns.

### Stage 2 — Tile Encoding with a Pre-trained Encoder

Each tile is independently encoded by a frozen (or lightly fine-tuned) foundation model.
The patch embedding trick is happening here, inside each tile's forward pass — but now
each "image" is only 256×256 px, so the sequence length is small (\(N=256\) for a ViT-B/16).

```python
# Standard tile encoding loop (batched for throughput)
import torch
from torchvision import transforms

model = load_pretrained_encoder("uni")  # or "conch", "pathdino", "ctranspath"
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

features = []
with torch.no_grad():
    for batch in DataLoader(tiles, batch_size=256):
        feat = model(batch)        # (batch, D)  — CLS token or mean pool
        features.append(feat.cpu())

slide_bag = torch.cat(features)    # (N_tiles, D) — the "bag of features"
torch.save(slide_bag, "slide_id.pt")  # cache — never recompute
```

**Key encoders used in the field (2024–2025):**

| Model | Architecture | Pretraining | Notes |
|---|---|---|---|
| UNI | ViT-L/16 | 100K+ slides, DINOv2 | General pathology; strongest transfer |
| CONCH | ViT-B/16 | Image–text pairs (CoCa) | Good for zero-shot + retrieval |
| PathDINO | ViT-S/8 | TCGA + GTEx, DINO | Lightweight, open weights |
| PLIP | ViT-B/32 (CLIP) | Twitter pathology captions | Text-image alignment |
| CTransPath | Swin-T hybrid | TCGA + PAIP | Older but widely cited |

### Stage 3 — Slide-Level Aggregation

The output of Stage 2 is a **bag** of tile features: a set \(\{f_1, \ldots, f_N\}\) with
no guaranteed ordering. The aggregation model must produce a single slide-level prediction
from this variable-length set. Three main approaches:

#### 3a. Attention-Based MIL (ABMIL) — the workhorse

Multiple Instance Learning (MIL) treats the slide as a *bag* and each tile as an *instance*.
ABMIL (Ilse et al., 2018) computes a weighted average where the weights are learned attention
scores:

$$\hat{h} = \sum_{i=1}^{N} a_i \cdot f_i, \quad a_i = \frac{\exp\!\left(\mathbf{w}^T \tanh(\mathbf{V} f_i)\right)}{\sum_j \exp\!\left(\mathbf{w}^T \tanh(\mathbf{V} f_j)\right)}$$

The attention weights \(a_i\) directly indicate which tiles are most predictive — highly
interpretable.

```python
class ABMIL(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.attention_V = nn.Linear(in_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        # bag: (N, D)
        scores = self.attention_w(torch.tanh(self.attention_V(bag)))  # (N, 1)
        weights = torch.softmax(scores, dim=0)                        # (N, 1)
        slide_repr = (weights * bag).sum(dim=0, keepdim=True)         # (1, D)
        return self.classifier(slide_repr)
```

ABMIL is the standard baseline. It is fast (processes bags of 10,000+ tiles in seconds)
and interpretable (the attention map highlights tumour regions).

#### 3b. Transformer Aggregator (HIPT, TransMIL)

Rather than a simple weighted mean, apply a small transformer over the bag of tile embeddings.
This allows tile–tile interactions — e.g., the model can reason about spatial co-occurrence
of patterns.

**HIPT** (Hierarchical Image Pyramid Transformer, Chen et al., 2022) builds a two-level
hierarchy:

```
256×256 px tiles → ViT-S/8 → 256-dim features (local)
     ↓  arrange into 4096×4096 px regions
4096-px regions → ViT-S/16 → region features (global)
     ↓
Slide-level aggregation head
```

**TransMIL** uses a Nyström-approximated transformer (linear complexity) to model
spatial and morphological correlations across tiles.

These models outperform ABMIL on complex tasks (survival prediction, mutation status),
at the cost of longer training and the need for more slides.

#### 3c. Mamba (State Space Model) Aggregator

Mamba (Gu & Dao, 2023) is a selective state space model with \(O(N)\) complexity and
strong performance on long sequences. Recent work (MambaMIL, 2024) applies Mamba as the
aggregation backbone, processing tiles in a sequence ordered by spatial proximity. It matches
or exceeds transformer aggregators while being 2–5× faster on long bags.

**When to use each aggregation approach:**

| Method | \(N\) range | Interpretability | When to choose |
|---|---|---|---|
| ABMIL | 100–50,000 | High (tile weights) | Default starting point |
| TransMIL / HIPT | 100–10,000 | Medium | Complex morphological tasks |
| Mamba (MambaMIL) | 1,000–100,000 | Low | Very large slides, speed-critical |
| Simple mean pool | Any | None | Sanity-check baseline only |

---

## Summary: Decision Tree

```
High-resolution image task?
│
├─ Standard CV (detection, segmentation)?
│   └─ Use Swin Transformer backbone
│       + Flash Attention (always on CUDA)
│
├─ Mobile / edge deployment?
│   └─ CNN stem hybrid (MobileViT, EfficientViT)
│
├─ Image classification, moderate resolution (≤512 px)?
│   └─ DINOv2 fine-tune or plain ViT + Flash Attention
│
└─ Medical / pathology / satellite (very high res)?
    ├─ 2D/3D radiology (MRI, CT)?
    │   └─ SwinUNETR or nnFormer (3D Conv patch embedding)
    │
    └─ Gigapixel (WSI, satellite)?
        └─ Tiling pipeline:
           Stage 1: tile extraction (256–512 px tiles)
           Stage 2: tile encoding with frozen foundation model
           Stage 3: ABMIL (default) or TransMIL/Mamba (complex tasks)
```

---

## References

- Dosovitskiy et al. (2021). [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929). ICLR 2021.
- Liu et al. (2021). [Swin Transformer](https://arxiv.org/abs/2103.14030). ICCV 2021.
- Liu et al. (2022). [Swin Transformer V2](https://arxiv.org/abs/2111.09883). CVPR 2022.
- Dao et al. (2022). [FlashAttention](https://arxiv.org/abs/2205.14135). NeurIPS 2022.
- Dao (2023). [FlashAttention-2](https://arxiv.org/abs/2307.08691).
- Ilse et al. (2018). [Attention-based Deep MIL](https://arxiv.org/abs/1802.04712). ICML 2018.
- Chen et al. (2022). [HIPT](https://arxiv.org/abs/2206.02647). CVPR 2022.
- Chen et al. (2024). [UNI](https://arxiv.org/abs/2308.15474). Nature Medicine 2024.
- Lu et al. (2021). [CLAM](https://arxiv.org/abs/2004.09666). Nature Biomedical Engineering 2021.
- Yang et al. (2024). [MambaMIL](https://arxiv.org/abs/2408.04427).
- Gu & Dao (2023). [Mamba](https://arxiv.org/abs/2312.00752).
