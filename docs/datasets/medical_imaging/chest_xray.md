# Chest X-Ray Datasets

This document covers chest X-ray datasets available in genai-lab for training diffusion models and other generative architectures.

---

## 1. Synthetic X-Ray Dataset

**Purpose**: Testing and development without requiring real medical data.

### Description

`SyntheticXRayDataset` generates diverse synthetic chest X-ray images with randomized anatomical structures:

- **Lungs**: Elliptical shapes with variable position, size, darkness
- **Heart**: Circular structure between lungs
- **Ribs**: Curved horizontal lines with variable spacing
- **Spine**: Optional vertical structure
- **Clavicles**: Optional horizontal structures at top

### Usage

```python
from genailab.diffusion.datasets import SyntheticXRayDataset

# Create dataset
dataset = SyntheticXRayDataset(
    n_samples=1000,    # Number of images to generate
    img_size=128,      # Image dimensions (square)
    seed=42            # Reproducibility
)

# Access images
img = dataset[0]  # Returns tensor of shape (1, 128, 128) in [-1, 1]
```

### Characteristics

| Property | Value |
|----------|-------|
| Output shape | `(1, H, W)` grayscale |
| Value range | `[-1, 1]` (normalized) |
| Generation | On-init (all images pre-generated) |
| Randomization | Per-image anatomical variation |

### When to Use

- **Development**: Fast iteration without data download
- **Testing**: Verify pipeline correctness
- **Debugging**: Reproducible synthetic data
- **Demos**: Quick demonstrations

---

## 2. Real Chest X-Ray Dataset (Kaggle)

**Purpose**: Training on real medical images for realistic diffusion models.

### Source

**Kaggle Chest X-Ray Images (Pneumonia)**

- URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- License: CC BY 4.0
- Size: ~1.2 GB

### Dataset Structure

```
chest_xray/
├── train/
│   ├── NORMAL/      (~1,300 images)
│   └── PNEUMONIA/   (~3,900 images)
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Usage

```python
from genailab.diffusion.datasets import ChestXRayDataset

# Load real X-rays (normal cases)
dataset = ChestXRayDataset(
    root_dir="data/chest_xray/train/NORMAL",
    img_size=128
)

# Or use factory function
from genailab.diffusion.datasets import get_dataset

dataset = get_dataset(
    name='chest_xray',
    root_dir="data/chest_xray/train/NORMAL",
    img_size=128
)
```

### Download Instructions

1. Create Kaggle account and API token
2. Install kaggle CLI: `pip install kaggle`
3. Download:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

### Characteristics

| Property | Value |
|----------|-------|
| Original resolution | Variable (typically 1000-2000 px) |
| Output shape | `(1, H, W)` grayscale (resized) |
| Value range | `[-1, 1]` (normalized) |
| Classes | NORMAL, PNEUMONIA |

---

## 3. Factory Function

Use `get_dataset()` for unified access:

```python
from genailab.diffusion.datasets import get_dataset

# Synthetic
synthetic_ds = get_dataset('synthetic', img_size=128, n_samples=1000)

# Real
real_ds = get_dataset('chest_xray', root_dir='data/chest_xray/train/NORMAL', img_size=128)
```

---

## 4. Integration with Training

### With Diffusion Training

```python
from torch.utils.data import DataLoader
from genailab.diffusion.datasets import get_dataset
from genailab.diffusion.training import train_image_diffusion

# Create dataset and loader
dataset = get_dataset('synthetic', img_size=64, n_samples=500)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train diffusion model
model, losses = train_image_diffusion(
    dataloader=dataloader,
    img_size=64,
    n_epochs=50,
    device='cuda'
)
```

### See Also

- `notebooks/diffusion/03_medical_imaging_diffusion/` — Full tutorial

---

## 5. Extending with New Datasets

To add a new medical imaging dataset:

1. Create a new `Dataset` class in `src/genailab/diffusion/datasets.py`
2. Implement `__len__` and `__getitem__`
3. Ensure output is `(C, H, W)` tensor in `[-1, 1]`
4. Add to `get_dataset()` factory function
5. Document here

---

## Related Code

| File | Contents |
|------|----------|
| `src/genailab/diffusion/datasets.py` | `SyntheticXRayDataset`, `ChestXRayDataset`, `get_dataset` |
| `src/genailab/diffusion/training.py` | `train_image_diffusion` |
| `src/genailab/diffusion/architectures.py` | `UNet2D` for image diffusion |
