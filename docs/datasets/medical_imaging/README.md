# Medical Imaging Datasets

Datasets for training diffusion models and other generative models on medical images.

---

## Available Datasets

| Dataset | Type | Size | Resolution | Document |
|---------|------|------|------------|----------|
| Synthetic X-Ray | Generated | Configurable | Configurable | [chest_xray.md](chest_xray.md) |
| Kaggle Chest X-Ray | Real | ~5,800 | Variable | [chest_xray.md](chest_xray.md) |

---

## Use Cases

Medical imaging datasets in genai-lab are used for:

1. **Diffusion model development**: Testing score networks, noise schedules, sampling
2. **DiT training**: Demonstrating Transformer-based diffusion
3. **Flow matching**: Rectified flow on realistic image data
4. **Benchmarking**: Comparing architectures on structured data

---

## Related Code

| Module | Purpose |
|--------|---------|
| `src/genailab/diffusion/datasets.py` | `SyntheticXRayDataset`, `ChestXRayDataset` |
| `src/genailab/diffusion/architectures.py` | `UNet2D`, `UNet3D` for image diffusion |
| `src/genailab/diffusion/training.py` | `train_image_diffusion` |

## Related Notebooks

- `notebooks/diffusion/03_medical_imaging_diffusion/` — Full pipeline demo

---

## Documents

- [chest_xray.md](chest_xray.md) — Chest X-ray datasets (synthetic and real)
