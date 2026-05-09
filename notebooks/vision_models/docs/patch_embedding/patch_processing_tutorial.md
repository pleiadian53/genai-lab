# Patch Embedding: Handling Non-Divisible Image Sizes

Source notebook: `mini_vit_from_scratch.ipynb`
Topic: What happens when `image_size % patch_size != 0`, and the four strategies to deal with it.

---

## The Problem

In `PatchEmbedding`, we use `Conv2d(kernel_size=P, stride=P)` to slice the image into non-overlapping patches. PyTorch computes the spatial output size as:

$$
H_{\text{out}} = \left\lfloor \frac{H - P}{P} \right\rfloor + 1 = \left\lfloor \frac{H}{P} \right\rfloor
$$

When $H$ is a perfect multiple of $P$, this is clean division — 32 pixels with patch size 4 gives exactly 8 patches per side.

When it's not, the remainder pixels are **silently dropped**. A 230×230 image with $P = 16$ yields $\lfloor 230 / 16 \rfloor = 14$ patches per side, and the rightmost and bottom 6 pixels vanish without any error or warning. That's roughly 5% of the image gone.

---

## Strategy 1: Pad to the Next Multiple

The most common production approach. Add pixels around the border so the padded size is exactly divisible by $P$.

```python
import math
import torch.nn.functional as F

def pad_to_multiple(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pad H and W up to the next multiple of patch_size."""
    _, _, H, W = x.shape
    pad_h = (patch_size - H % patch_size) % patch_size  # 0 if already divisible
    pad_w = (patch_size - W % patch_size) % patch_size
    # F.pad order: (left, right, top, bottom)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
```

After padding, `Conv2d(kernel_size=P, stride=P)` works exactly. The padded pixels contribute to border patches, but the model learns to ignore zeros in practice.

**Padding mode choices:**

| Mode | What fills the border | When to use |
|---|---|---|
| `constant` (value=0) | Black pixels | Most common, simplest |
| `reflect` | Mirror of real pixels | Natural images, reduces border artefacts |
| `replicate` | Edge pixel repeated | Similar to reflect |

---

## Strategy 2: Resize to the Nearest Valid Resolution

Resize the image so both spatial dims are multiples of $P$:

```python
import torchvision.transforms.functional as TF

def resize_to_multiple(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, _, H, W = x.shape
    H_new = round(H / patch_size) * patch_size
    W_new = round(W / patch_size) * patch_size
    return TF.resize(x, [H_new, W_new])
```

This introduces slight spatial distortion (interpolation) but avoids injecting artificial padding values into the patches. The tradeoff is content fidelity vs. clean boundaries — for most natural image tasks, this distortion is negligible.

---

## Strategy 3: Overlapping Patches via Sliding Window

Instead of non-overlapping strides, use `torch.Tensor.unfold` to extract patches with a stride smaller than the patch size. This produces more patches and covers every pixel:

```python
def extract_patches(
    x: torch.Tensor, patch_size: int, stride: int
) -> torch.Tensor:
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    patches = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # patches: (B, C, n_h, n_w, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    return patches  # (B, C, N, P, P)
```

The sequence length $N$ increases, and patches share pixels at the overlap regions. This is standard in tiling pipelines — medical imaging (whole-slide images) and satellite imagery — where losing any region is unacceptable.

---

## Strategy 4: Fractional / Learned Patch Size

Some architectures make patch size a runtime variable. **FlexViT** (Google, 2022) dynamically resizes the projection weights via interpolation, letting a single model handle any resolution at inference:

```python
# FlexViT-style: interpolate the projection weight to match actual patch size
weight = F.interpolate(
    self.projection.weight,
    size=(actual_patch_h, actual_patch_w),
    mode="bilinear",
    align_corners=False,
)
x = F.conv2d(x, weight, stride=(actual_patch_h, actual_patch_w))
```

This is the most flexible option but adds complexity — the Conv2d kernel shape changes per forward pass, so you lose the benefit of fixed-shape GEMM optimization.

---

## What Production Systems Actually Do

| System | Strategy |
|---|---|
| Original ViT (ImageNet) | Fixed resize to 224×224 before patching; always divisible |
| DINOv2 | Resize + reflect pad to nearest multiple of 14 ($P = 14$) |
| Swin Transformer | Pad to multiple of window size (32 or 64 px) with constant |
| SAM (Segment Anything) | Pad to 1024×1024 exactly with constant padding |
| SwinUNETR (medical) | 3D pad to multiple of (2,2,2) or (4,4,4) |
| FlexViT | Fractional patch size, no resizing needed |
| Tiling pipelines (WSI) | Extract non-overlapping 256 px tiles, discard partial border tiles |

The dominant production pattern: **resize first, then pad to multiple**. Resize to a canonical resolution (224, 384, 512, etc.) during preprocessing, then add reflect/constant padding as a safety net. This keeps sequence length $N$ predictable and allows batching of images with the same shape.

---

## Practical Extension for the Notebook

The cleanest way to make `PatchEmbedding` handle arbitrary input sizes — replace the hard `assert` with a pad-to-multiple in `forward`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    _, _, H, W = x.shape
    pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
    pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right and bottom only
    x = self.projection(x)
    x = x.flatten(2).transpose(1, 2)
    return x
```

This removes the assert, handles arbitrary input sizes gracefully, and adds zero overhead when the image is already a perfect multiple. One subtlety: if you do this, `num_patches` is no longer a fixed attribute — it depends on the input shape. Positional embeddings would need to be interpolated to match (which is exactly what DINOv2 and the original ViT fine-tuning pipeline do when changing resolution).
