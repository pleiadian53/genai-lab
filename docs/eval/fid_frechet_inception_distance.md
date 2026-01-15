# Fréchet Inception Distance (FID): Theory and Practice

Fréchet Inception Distance (FID) is the most widely used metric for evaluating generative models, particularly for image generation. It measures the distance between the distribution of generated samples and real samples in a learned feature space.

This document explains the mathematical foundations, practical computation, interpretation, and limitations of FID.

---

## Overview

### What FID Measures

FID quantifies how similar two distributions are by comparing their statistics in a high-dimensional feature space:

- **Real distribution**: Features extracted from real training data
- **Generated distribution**: Features extracted from model-generated samples

**Key insight**: Rather than comparing raw pixels, FID compares distributions in a semantically meaningful feature space learned by a pre-trained neural network.

### Why FID is Popular

1. **Captures both quality and diversity**: Unlike metrics that measure only one aspect
2. **Correlates with human judgment**: Better than earlier metrics like Inception Score
3. **Robust and stable**: Less sensitive to small sample variations than IS
4. **Widely adopted**: Standard benchmark across papers and models
5. **Single scalar**: Easy to compare models

---

## Mathematical Foundation

### The Fréchet Distance

FID is based on the **Fréchet distance** (also called Wasserstein-2 distance) between two multivariate Gaussian distributions.

Given two Gaussian distributions:
- $\mathcal{N}(\mu_r, \Sigma_r)$ for real data
- $\mathcal{N}(\mu_g, \Sigma_g)$ for generated data

The Fréchet distance is:

$$
d^2(\mathcal{N}(\mu_r, \Sigma_r), \mathcal{N}(\mu_g, \Sigma_g)) = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

where:

- $\mu_r, \mu_g \in \mathbb{R}^d$ are mean vectors
- $\Sigma_r, \Sigma_g \in \mathbb{R}^{d \times d}$ are covariance matrices
- $\text{Tr}(\cdot)$ is the matrix trace
- $(\Sigma_r \Sigma_g)^{1/2}$ is the matrix square root of the product

### Intuition

The FID formula has two terms:

**1. Mean difference**: $\|\mu_r - \mu_g\|^2$
- Measures whether generated samples have the same "center" as real samples
- Captures whether the model generates the right kind of content on average

**2. Covariance difference**: $\text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$
- Measures whether generated samples have the same "spread" and correlations as real samples
- Captures diversity and feature relationships

**Lower FID = Better**: Smaller distance means distributions are more similar.

---

## Computation Pipeline

### Step 1: Feature Extraction

Extract features from a pre-trained network (typically **Inception-v3**):

1. **Load pre-trained Inception-v3**: Trained on ImageNet
2. **Extract features**: Use the final pooling layer (before classification)
   - Output: 2048-dimensional feature vector per image
3. **Process all images**:
   - Real images: $\{x_1^r, x_2^r, \ldots, x_N^r\}$
   - Generated images: $\{x_1^g, x_2^g, \ldots, x_M^g\}$

**Feature extraction**:

$$
f_i^r = \text{Inception}(x_i^r) \in \mathbb{R}^{2048}
$$

$$
f_j^g = \text{Inception}(x_j^g) \in \mathbb{R}^{2048}
$$

### Step 2: Compute Statistics

Calculate mean and covariance for both distributions:

**Real data statistics**:

$$
\mu_r = \frac{1}{N} \sum_{i=1}^N f_i^r
$$

$$
\Sigma_r = \frac{1}{N-1} \sum_{i=1}^N (f_i^r - \mu_r)(f_i^r - \mu_r)^T
$$

**Generated data statistics**:

$$
\mu_g = \frac{1}{M} \sum_{j=1}^M f_j^g
$$

$$
\Sigma_g = \frac{1}{M-1} \sum_{j=1}^M (f_j^g - \mu_g)(f_j^g - \mu_g)^T
$$

### Step 3: Compute FID

Calculate the Fréchet distance:

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

**Matrix square root computation**:

The term $(\Sigma_r \Sigma_g)^{1/2}$ requires computing the matrix square root:

1. Compute eigendecomposition: $\Sigma_r \Sigma_g = Q \Lambda Q^T$
2. Take square root: $(\Sigma_r \Sigma_g)^{1/2} = Q \Lambda^{1/2} Q^T$

**Numerical stability**: Add small epsilon to diagonal for numerical stability:

$$
\Sigma_r \leftarrow \Sigma_r + \epsilon I, \quad \epsilon \approx 10^{-6}
$$

---

## Practical Implementation

### Python Example

```python
import numpy as np
from scipy import linalg
from torch import nn
from torchvision.models import inception_v3

def calculate_fid(real_images, generated_images, inception_model):
    """
    Calculate FID between real and generated images.
    
    Args:
        real_images: Tensor of real images [N, C, H, W]
        generated_images: Tensor of generated images [M, C, H, W]
        inception_model: Pre-trained Inception-v3 model
    
    Returns:
        fid_score: Scalar FID value
    """
    # Extract features
    with torch.no_grad():
        features_real = inception_model(real_images).cpu().numpy()
        features_gen = inception_model(generated_images).cpu().numpy()
    
    # Compute statistics
    mu_real = np.mean(features_real, axis=0)
    mu_gen = np.mean(features_gen, axis=0)
    
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_gen = np.cov(features_gen, rowvar=False)
    
    # Compute FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet distance between two Gaussian distributions.
    """
    # Mean difference
    diff = mu1 - mu2
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical stability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Handle complex numbers from numerical errors
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    # Compute FID
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    
    return fid
```

### Using Existing Libraries

**PyTorch FID** (recommended):

```python
from pytorch_fid import fid_score

# Compute FID between two directories of images
fid_value = fid_score.calculate_fid_given_paths(
    [path_to_real_images, path_to_generated_images],
    batch_size=50,
    device='cuda',
    dims=2048
)
print(f'FID: {fid_value:.2f}')
```

**Clean-FID** (improved implementation):

```python
from cleanfid import fid

# More robust FID computation
fid_value = fid.compute_fid(
    path_to_real_images,
    path_to_generated_images,
    mode="clean",
    num_workers=4
)
```

---

## Interpretation and Typical Values

### What Different FID Values Mean

| FID Range | Interpretation | Example |
|-----------|----------------|---------|
| 0 | Perfect match | Comparing dataset to itself |
| < 10 | Excellent quality | SOTA diffusion models on ImageNet |
| 10-30 | Good quality | Strong generative models |
| 30-50 | Moderate quality | Early GANs, simple models |
| 50-100 | Poor quality | Weak models, mode collapse |
| > 100 | Very poor quality | Random noise, severe artifacts |

### Benchmark Values

**ImageNet 256×256** (as of 2024):
- **Real data**: FID = 0 (by definition)
- **SOTA diffusion models**: FID ≈ 2-5
- **Good GANs**: FID ≈ 10-20
- **Early GANs**: FID ≈ 50-100

**CIFAR-10**:

- **SOTA models**: FID ≈ 2-3
- **Good models**: FID ≈ 5-10
- **Moderate models**: FID ≈ 20-40

**CelebA-HQ**:

- **SOTA models**: FID ≈ 3-8
- **Good models**: FID ≈ 10-20

### FID is Relative

**Important**: FID values are **not comparable across datasets**:
- FID = 10 on ImageNet ≠ FID = 10 on CIFAR-10
- Always compare models on the **same dataset** with the **same evaluation protocol**

---

## Advantages of FID

### 1. Captures Quality and Diversity

Unlike Inception Score (which primarily measures quality), FID captures both:
- **Quality**: Via mean difference (are samples realistic?)
- **Diversity**: Via covariance difference (do samples cover the distribution?)

### 2. Robust to Sample Size

FID is relatively stable with sufficient samples:
- **Minimum recommended**: 10,000 samples (both real and generated)
- **Typical**: 50,000 samples for reliable estimates
- **Variance decreases** with more samples

### 3. Correlates with Human Judgment

Studies show FID correlates better with human perceptual quality than earlier metrics like IS.

### 4. Detects Mode Collapse

If a model generates only a subset of the distribution (mode collapse):
- Mean may be similar
- Covariance will differ significantly
- FID will be high

### 5. Differentiable (in principle)

While not typically used as a training loss, FID can be approximated for gradient-based optimization.

---

## Limitations and Pitfalls

### 1. Assumes Gaussian Distributions

**Assumption**: Features follow multivariate Gaussian distributions.

**Reality**: Feature distributions may not be Gaussian.

**Impact**: FID may not capture all distributional differences, especially in tails.

### 2. Requires Pre-trained Network

**Dependency**: Typically uses Inception-v3 trained on ImageNet.

**Issues**:

- **Domain mismatch**: ImageNet features may not be optimal for medical images, satellite imagery, etc.
- **Bias**: Inherits biases from ImageNet training
- **Not universal**: Doesn't work for non-image data without adaptation

### 3. Sample Size Sensitivity

**Problem**: FID estimates have variance that depends on sample size.

**Recommendations**:

- Use **same sample size** for all comparisons
- Use **≥10,000 samples** for stable estimates
- Report **multiple runs** with different random samples

**Example variance**:
```
N = 1,000:  FID = 15.3 ± 2.1
N = 10,000: FID = 14.8 ± 0.4
N = 50,000: FID = 14.7 ± 0.1
```

### 4. Doesn't Capture All Aspects

**What FID misses**:

- **Local structure**: Fine-grained details
- **Semantic correctness**: Object relationships, physical plausibility
- **Perceptual quality**: Some artifacts humans notice
- **Diversity within modes**: Can miss subtle diversity issues

### 5. Can Be Gamed

**Memorization**: A model that memorizes training data will have low FID but poor generalization.

**Solution**: Always check for overfitting with held-out test sets.

### 6. Computational Cost

**Requirements**:

- Pre-trained Inception-v3 model
- Feature extraction for all images
- Covariance computation (scales as $O(d^2 n)$ for $d$ dimensions, $n$ samples)

**Typical time**: Minutes to hours depending on dataset size.

---

## Best Practices

### 1. Standardize Evaluation Protocol

**Image preprocessing**:

- Resize to 299×299 (Inception-v3 input size)
- Normalize to [0, 1] or [-1, 1] consistently
- Use same preprocessing for real and generated images

**Sample size**:

- Use ≥10,000 samples (preferably 50,000)
- Use same number of real and generated samples
- Report sample size in papers

### 2. Multiple Runs

Compute FID with **multiple random seeds** or sample sets:

```python
fid_scores = []
for seed in range(5):
    generated_samples = model.sample(n=10000, seed=seed)
    fid = compute_fid(real_samples, generated_samples)
    fid_scores.append(fid)

mean_fid = np.mean(fid_scores)
std_fid = np.std(fid_scores)
print(f'FID: {mean_fid:.2f} ± {std_fid:.2f}')
```

### 3. Use Consistent Implementation

**Recommended**: Use standard libraries (pytorch-fid, clean-fid) rather than custom implementations.

**Why**: Subtle implementation differences can cause significant FID variations.

### 4. Report Full Details

**Minimum reporting**:

- FID value with standard deviation
- Number of samples used
- Feature extractor (Inception-v3, etc.)
- Image resolution
- Preprocessing steps

### 5. Complement with Other Metrics

**Never rely on FID alone**. Use complementary metrics:
- **Precision/Recall**: Quality vs. diversity trade-off
- **IS**: Additional quality measure
- **Human evaluation**: Perceptual quality
- **Task-specific metrics**: Downstream performance

### 6. Domain-Specific Adaptations

For non-natural images:
- Consider using domain-specific feature extractors
- Train custom feature networks on your domain
- Validate that features capture meaningful semantics

---

## Variants and Extensions

### Clean-FID

**Improvement**: More robust implementation addressing numerical issues.

**Changes**:

- Better handling of edge cases
- Improved numerical stability
- Consistent preprocessing

**Usage**: Recommended over original implementation.

### Kernel FID (KID)

**Alternative**: Uses kernel methods instead of Gaussian assumption.

**Advantages**:

- No Gaussian assumption
- Unbiased estimator
- Better for small sample sizes

**Formula**:

$$
\text{KID} = \mathbb{E}[k(x_r, x_r')] + \mathbb{E}[k(x_g, x_g')] - 2\mathbb{E}[k(x_r, x_g)]
$$

where $k(\cdot, \cdot)$ is a kernel function (e.g., polynomial kernel).

### Precision and Recall

**Decomposition**: Separates FID into quality (precision) and diversity (recall).

**Advantages**:

- More interpretable
- Detects mode collapse vs. poor quality
- Complements FID

### Domain-Specific FID

**Adaptations**:

- **Medical imaging**: Use pre-trained medical image networks
- **Satellite imagery**: Use remote sensing features
- **Biological data**: Use domain-specific embeddings

---

## FID for Non-Image Data

### Adapting FID to Other Domains

**General approach**:
1. Define a meaningful feature space
2. Extract features from real and generated data
3. Compute Fréchet distance

### Examples

**Text**:

- Use BERT or GPT embeddings as features
- Compute FID in embedding space

**Audio**:

- Use audio feature extractors (e.g., VGGish)
- Compute FID on spectrograms or learned features

**Molecular structures**:

- Use molecular fingerprints or graph embeddings
- Compute FID in chemical space

**Gene expression**:

- Use pathway embeddings or PCA features
- Compute FID in biological feature space

**Caution**: Validate that chosen features capture meaningful semantics for your domain.

---

## Comparison with Other Metrics

### FID vs. Inception Score (IS)

| Aspect | FID | IS |
|--------|-----|-----|
| Requires real data | Yes | No |
| Captures diversity | Yes | Limited |
| Detects mode collapse | Yes | No |
| Robust to sample size | More robust | Less robust |
| Computational cost | Higher | Lower |
| Correlation with quality | Better | Good |

**Recommendation**: Use FID as primary metric, IS as secondary.

### FID vs. Precision/Recall

| Aspect | FID | Precision/Recall |
|--------|-----|------------------|
| Single number | Yes | Two numbers |
| Interpretability | Lower | Higher |
| Quality vs. diversity | Combined | Separated |
| Computational cost | Lower | Higher |

**Recommendation**: Use both—FID for overall quality, P/R for diagnosing issues.

### FID vs. Likelihood

| Aspect | FID | Likelihood |
|--------|-----|------------|
| Perceptual quality | Good | Poor |
| Theoretical grounding | Heuristic | Principled |
| Requires tractable model | No | Yes |
| Captures imperceptible details | No | Yes |

**Recommendation**: FID for perceptual quality, likelihood for distributional fit.

---

## Summary

### Key Takeaways

1. **FID measures distribution distance** in learned feature space
2. **Lower is better**: FID = 0 means perfect match
3. **Captures quality and diversity**: Both mean and covariance matter
4. **Widely adopted**: Standard benchmark for generative models
5. **Has limitations**: Gaussian assumption, domain dependence, sample size sensitivity

### When to Use FID

**Use FID when**:

- Evaluating image generative models
- Comparing different models on same dataset
- Need single scalar metric
- Have sufficient samples (≥10,000)

**Don't rely solely on FID when**:

- Working with non-image data (adapt carefully)
- Need to diagnose specific issues (use P/R)
- Sample size is small (<1,000)
- Domain is very different from ImageNet

### Best Practice Summary

✅ Use ≥10,000 samples
✅ Report mean ± std over multiple runs
✅ Use standard implementations (pytorch-fid, clean-fid)
✅ Complement with other metrics (P/R, IS)
✅ Report full evaluation details
✅ Compare on same dataset with same protocol

---

## Related Documents

- [Evaluating Generative Models](00_evaluating_generative_models.md) — Comprehensive evaluation guide
- [Epiplexity: From Entropy to Epiplexity](epiplexity/01_from_entropy_to_epiplexity.md) — Alternative evaluation perspective
- [DDPM Foundations](../DDPM/01_ddpm_foundations.md) — Diffusion model theory
- [DDPM Sampling](../DDPM/03_ddpm_sampling.md) — Sampling efficiency vs. quality

---

## References

### Original Paper
1. **Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017)**. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *NeurIPS*. [Paper](https://arxiv.org/abs/1706.08500)

### Implementations
2. **pytorch-fid**: [GitHub](https://github.com/mseitzer/pytorch-fid)
3. **clean-fid**: [GitHub](https://github.com/GaParmar/clean-fid) | [Paper](https://arxiv.org/abs/2104.11222)

### Related Metrics
4. **Salimans, T., et al. (2016)**. Improved Techniques for Training GANs. *NeurIPS*. (Inception Score)
5. **Kynkäänniemi, T., et al. (2019)**. Improved Precision and Recall Metric for Assessing Generative Models. *NeurIPS*.
6. **Bińkowski, M., et al. (2018)**. Demystifying MMD GANs. *ICLR*. (Kernel Inception Distance)

### Analysis and Best Practices
7. **Chong, M. J., & Forsyth, D. (2020)**. Effectively Unbiased FID and Inception Score and Where to Find Them. *CVPR*.
8. **Parmar, G., et al. (2021)**. On Aliased Resizing and Surprising Subtleties in GAN Evaluation. *CVPR*. (Clean-FID)
