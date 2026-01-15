# Medical Imaging Diffusion Models

This notebook demonstrates diffusion models on **realistic medical images**, bridging the gap between toy examples (Swiss roll) and high-dimensional applications (gene expression).

## Learning Objectives

1. Apply U-Net architecture to real medical images
2. Handle grayscale medical imaging data (X-rays, CT slices)
3. Implement data preprocessing for medical images
4. Generate synthetic medical images with diffusion models
5. Evaluate quality with domain-specific metrics

## Datasets Used

We use **publicly available, realistic medical imaging datasets** that are computationally feasible:

### 1. **Chest X-Ray Images (Primary Dataset)**
- **Source**: NIH Chest X-ray Dataset (downsampled)
- **Size**: 128×128 grayscale images (downsampled from 1024×1024)
- **Modality**: X-ray (radiography)
- **Use case**: Generate synthetic chest X-rays
- **Why**: Widely used, clinically relevant, single-channel (memory efficient)
- **Download**: Available via Kaggle or NIH Clinical Center

### 2. **Brain MRI Slices (Alternative)**
- **Source**: BraTS or IXI Dataset (2D slices)
- **Size**: 128×128 or 256×256 grayscale
- **Modality**: MRI (T1, T2, FLAIR)
- **Use case**: Generate brain MRI slices
- **Why**: Important for neuroimaging, good for demonstrating multi-modal generation

### 3. **Histopathology Patches (Advanced)**
- **Source**: Camelyon16/17 or PatchCamelyon
- **Size**: 96×96 RGB patches
- **Modality**: H&E stained tissue
- **Use case**: Generate tissue patches for data augmentation
- **Why**: Connects to your pathology-ai-lab project

## Computational Requirements

### Memory-Efficient Setup (Recommended)
- **Image size**: 128×128 (or 64×64 for faster iteration)
- **Batch size**: 16-32
- **Model**: UNet2D with base_channels=32 or 64
- **Training time**: 2-4 hours on M1/M2 Mac or consumer GPU
- **Memory**: ~4-8GB GPU/unified memory

### Full-Resolution Setup (If resources available)
- **Image size**: 256×256
- **Batch size**: 8-16
- **Model**: UNet2D with base_channels=64
- **Training time**: 8-12 hours
- **Memory**: ~16GB GPU memory

## Notebook Structure

1. **Setup & Data Loading**
   - Download and preprocess medical images
   - Create PyTorch dataset
   - Visualize samples

2. **Model Architecture**
   - Implement UNet2D for medical images
   - Time conditioning
   - GroupNorm for small batches

3. **Training**
   - VP-SDE with cosine schedule
   - Score matching loss
   - Training loop with checkpointing

4. **Generation & Evaluation**
   - Sample synthetic images
   - Visual quality assessment
   - Quantitative metrics (FID, IS)
   - Domain-specific evaluation

5. **Applications**
   - Data augmentation for downstream tasks
   - Conditional generation (by disease, view angle)
   - Inpainting and super-resolution

## Key Differences from Toy Examples

| Aspect | Toy (Swiss Roll) | Medical Imaging |
|--------|------------------|-----------------|
| **Data** | 2D points | 128×128 images (16K dims) |
| **Architecture** | Simple MLP | U-Net with skip connections |
| **Training time** | Minutes | Hours |
| **Evaluation** | Visual | FID, clinical metrics |
| **Applications** | Educational | Data augmentation, synthesis |

## Prerequisites

- Completed `02_sde_formulation.ipynb`
- Understanding of convolutional neural networks
- Familiarity with medical imaging (helpful but not required)

## Next Steps

After this notebook:
- `04_gene_expression_diffusion.ipynb`: High-dimensional tabular data
- Your `pathology-ai-lab` project: Whole-slide imaging with diffusion models

## References

### Datasets
- **NIH Chest X-ray**: Wang et al. (2017) "ChestX-ray8: Hospital-scale Chest X-ray Database"
- **BraTS**: Menze et al. (2015) "The Multimodal Brain Tumor Image Segmentation Benchmark"
- **Camelyon**: Bejnordi et al. (2017) "Diagnostic Assessment of Deep Learning Algorithms"

### Medical Imaging Diffusion
- **MedSegDiff**: Wu et al. (2023) "MedSegDiff: Medical Image Segmentation with Diffusion Models"
- **DiffMIC**: Özbey et al. (2023) "Unsupervised Medical Image Translation with Adversarial Diffusion Models"
- **RoentGen**: Chambon et al. (2022) "RoentGen: Vision-Language Foundation Model for Chest X-ray Generation"

### Architecture
- **U-Net**: Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **DDPM**: Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
