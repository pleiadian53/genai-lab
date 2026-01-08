# Diffusion Model Training Scripts

This directory contains production-ready scripts for training diffusion models.

## Scripts

### 1. Medical Imaging Diffusion (`03_medical_imaging_diffusion.py`)

Train diffusion models on medical imaging data (chest X-rays).

```bash
# Quick test with tiny model
python 03_medical_imaging_diffusion.py --preset tiny --epochs 10

# Full training with large model (requires GPU)
python 03_medical_imaging_diffusion.py --preset large --epochs 5000 --sample
```

**Model Presets:**

| Preset | Parameters | VRAM | Recommended Hardware |
|--------|------------|------|---------------------|
| tiny   | ~1M        | <2GB | Any (CPU OK)        |
| small  | ~5M        | ~4GB | T4, RTX 3060        |
| medium | ~20M       | ~8GB | RTX 3080, A10       |
| large  | ~50M       | ~16GB| A40, A100           |

### 2. Chest X-ray Data Pipeline (`chest_xray_data_pipeline.py`)

Download, analyze, and preprocess chest X-ray datasets from Kaggle.

#### Prerequisites

1. Install Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Setup credentials:
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings)
   - Scroll to 'API' section → 'Create New Token'
   - Save `kaggle.json` to `~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

#### Commands

```bash
# Download pneumonia dataset (~1.3GB)
python chest_xray_data_pipeline.py download --dataset pneumonia

# Analyze dataset statistics
python chest_xray_data_pipeline.py analyze --data-dir data/chest_xray

# Visualize sample images
python chest_xray_data_pipeline.py visualize --data-dir data/chest_xray -n 16

# Preprocess for training (resize to 128x128)
python chest_xray_data_pipeline.py preprocess \
    --data-dir data/chest_xray \
    --output-dir data/chest_xray_128 \
    --size 128

# Test PyTorch DataLoader
python chest_xray_data_pipeline.py test --data-dir data/chest_xray
```

#### Available Datasets

| Dataset | Size | Images | Description |
|---------|------|--------|-------------|
| `pneumonia` | ~1.3GB | 5,863 | Chest X-Ray Images (Pneumonia) - Normal vs Pneumonia |
| `nih` | ~45GB | 112,120 | NIH Chest X-rays - 14 disease labels |

## Workflow

### Option A: Synthetic Data (Quick Start)

```bash
# Train on synthetic X-rays (no download needed)
python 03_medical_imaging_diffusion.py --preset tiny --epochs 100
```

### Option B: Real Kaggle Data

```bash
# 1. Download dataset
python chest_xray_data_pipeline.py download --dataset pneumonia

# 2. Analyze and visualize
python chest_xray_data_pipeline.py analyze --data-dir data/chest_xray
python chest_xray_data_pipeline.py visualize --data-dir data/chest_xray

# 3. Preprocess
python chest_xray_data_pipeline.py preprocess \
    --data-dir data/chest_xray \
    --output-dir data/chest_xray_128 \
    --size 128

# 4. Train (modify script to use ChestXRayDataset)
# See notebooks/diffusion/03_medical_imaging_diffusion/ for examples
```

## Output Structure

Training outputs are saved to `checkpoints/diffusion/medical_imaging/<run-name>/`:

```
checkpoints/diffusion/medical_imaging/tiny_20240108_143022/
├── config.json           # Training configuration
├── model_final.pt        # Final model checkpoint
├── losses.npy            # Training loss history
├── training_curve.png    # Loss plot
├── samples.npy           # Generated samples (if --sample)
└── generated_samples.png # Sample grid (if --sample)
```

## Related

- **Notebooks**: `notebooks/diffusion/03_medical_imaging_diffusion/` - Interactive tutorial
- **Source**: `src/genailab/diffusion/` - Core diffusion module
- **Docs**: `docs/runpods/` - GPU cloud setup for large-scale training
