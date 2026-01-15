# Evaluating Generative Models: A Comprehensive Guide

Evaluating generative models is fundamentally challenging because there is no single "ground truth" metric. Different evaluation approaches capture different aspects of model quality, and the choice of metrics depends on the application domain and goals.

This document provides a comprehensive overview of evaluation methods for generative models, with particular focus on diffusion models and biological data applications.

---

## The Evaluation Challenge

### Why Evaluation is Hard

Generative models face unique evaluation challenges:

1. **No single ground truth**: Unlike supervised learning, there's no single correct output
2. **Multiple objectives**: Quality, diversity, efficiency, and utility often trade off
3. **Domain-specific requirements**: What matters for images differs from what matters for gene expression
4. **Computational constraints**: Some metrics are expensive to compute
5. **Perceptual vs. statistical**: Human perception doesn't always align with statistical measures

### Key Questions

When evaluating a generative model, ask:

- **Quality**: Are individual samples realistic?
- **Diversity**: Does the model cover the full data distribution?
- **Fidelity**: Does the model match the training distribution?
- **Utility**: Is the generated data useful for downstream tasks?
- **Efficiency**: How much compute is required for training and sampling?
- **Controllability**: Can we guide generation toward desired outputs?

---

## Evaluation Metrics by Category

### 1. Sample Quality Metrics

#### Inception Score (IS)

**What it measures**: Quality and diversity of generated images

$$
\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g} \left[D_{KL}(p(y|x) \| p(y))\right]\right)
$$

where $p(y|x)$ is a pre-trained classifier's prediction.

**Interpretation**:

- Higher is better
- Good samples should have confident class predictions (low $p(y|x)$ entropy)
- Diverse samples should cover many classes (high $p(y)$ entropy)

**Limitations**:

- Requires pre-trained classifier (typically ImageNet-trained Inception network)
- Doesn't detect overfitting or mode collapse well
- Not applicable to non-image domains

**Typical values**:

- Random noise: ~1
- CIFAR-10 real data: ~11.2
- Good generative models: 8-10

#### Fréchet Inception Distance (FID)

**What it measures**: Distance between real and generated data distributions in feature space

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

where $\mu_r, \Sigma_r$ are mean and covariance of real data features, and $\mu_g, \Sigma_g$ for generated data.

**Interpretation**:

- Lower is better
- Measures both quality and diversity
- More robust than IS

**Limitations**:

- Requires pre-trained feature extractor
- Sensitive to sample size (need 10k+ samples)
- Assumes Gaussian distributions in feature space

**Typical values**:

- Perfect match: 0
- Excellent: <10
- Good: 10-30
- Poor: >100

#### Precision and Recall

**What it measures**: Quality vs. diversity trade-off

- **Precision**: Fraction of generated samples that are realistic
- **Recall**: Fraction of real distribution covered by generated samples

**Interpretation**:

- Precision measures quality (are generated samples realistic?)
- Recall measures diversity (does the model cover the full distribution?)
- High precision, low recall = mode collapse
- Low precision, high recall = poor quality but diverse

**Advantages**:

- Separates quality and diversity
- More interpretable than FID
- Detects mode collapse

### 2. Likelihood-Based Metrics

#### Negative Log-Likelihood (NLL)

**What it measures**: How well the model assigns probability to real data

$$
\text{NLL} = -\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
$$

**Interpretation**:

- Lower is better
- Directly measures distributional fit
- Theoretically principled

**Limitations**:

- Requires tractable likelihood (not available for GANs)
- Can be dominated by imperceptible details
- Doesn't measure perceptual quality

#### Evidence Lower Bound (ELBO)

**What it measures**: Lower bound on log-likelihood (for VAEs, diffusion models)

$$
\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p_\theta(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

**Interpretation**:

- Higher is better
- Balances reconstruction and regularization
- Tractable for many models

**Limitations**:

- Only a lower bound (gap to true likelihood unknown)
- May not correlate with sample quality

### 3. Perceptual Metrics

#### Learned Perceptual Image Patch Similarity (LPIPS)

**What it measures**: Perceptual distance between images using deep features

$$
\text{LPIPS}(x, x') = \sum_l w_l \|f_l(x) - f_l(x')\|^2
$$

where $f_l$ are features from layer $l$ of a pre-trained network.

**Interpretation**:

- Lower is better
- Correlates better with human perception than pixel-wise metrics
- Useful for image-to-image tasks

**Advantages**:

- Captures perceptual similarity
- More robust than MSE or SSIM

**Limitations**:

- Requires pre-trained network
- Domain-specific (primarily images)

### 4. Domain-Specific Metrics

#### For Images
- **SSIM**: Structural similarity
- **PSNR**: Peak signal-to-noise ratio
- **Human evaluation**: Perceptual studies

#### For Text
- **BLEU**: N-gram overlap with references
- **Perplexity**: Language model likelihood
- **Human evaluation**: Fluency, coherence

#### For Biological Data
- **Correlation with real data**: Gene-gene correlations
- **Pathway enrichment**: Biological pathway preservation
- **Cell type classification**: Downstream task performance
- **Perturbation response**: Causal structure preservation

---

## Evaluating Diffusion Models Specifically

### Standard Diffusion Model Metrics

#### 1. Sample Quality
- **FID**: Primary metric for image diffusion models
- **IS**: Secondary metric
- **Precision/Recall**: Trade-off analysis

#### 2. Likelihood
- **ELBO**: Variational bound
- **Bits per dimension (BPD)**: Normalized likelihood
  $$
  \text{BPD} = -\frac{\log_2 p(x)}{D}
  $$

  where $D$ is data dimensionality

#### 3. Sampling Efficiency
- **NFE (Number of Function Evaluations)**: How many denoising steps?
- **Wall-clock time**: Actual sampling time
- **FID vs. NFE trade-off**: Quality-speed curve

**Typical trade-offs**:

- DDPM: 1000 steps, high quality
- DDIM: 50-100 steps, good quality
- Fast samplers: 10-20 steps, acceptable quality

#### 4. Training Efficiency
- **Training time**: Hours/days to convergence
- **Compute cost**: GPU-hours
- **Sample efficiency**: Performance vs. dataset size

### Diffusion-Specific Considerations

#### Noise Schedule Evaluation
- **Loss curve shape**: Progressive learning vs. flat
- **Timestep importance**: Which timesteps matter most?
- **Schedule ablations**: Linear vs. cosine vs. learned

#### Guidance Strength
- **FID vs. guidance scale**: Quality-diversity trade-off
- **Classifier-free guidance**: Typical range $w \in [1, 10]$
- **Optimal guidance**: Task-dependent

---

## Comparing Modeling Approaches

### How to Determine "Better" or "Worse"

#### 1. Define Your Objectives

**For research**:

- State-of-the-art sample quality (FID)
- Theoretical contributions (likelihood bounds)
- Novel capabilities (controllability, efficiency)

**For applications**:

- Downstream task performance
- Computational efficiency
- Robustness and reliability

#### 2. Multi-Metric Evaluation

Never rely on a single metric. Use a **suite of complementary metrics**:

| Aspect | Metrics | Why |
|--------|---------|-----|
| Quality | FID, IS, Precision | Are samples realistic? |
| Diversity | Recall, Mode coverage | Full distribution? |
| Fidelity | Likelihood, ELBO | Statistical match? |
| Utility | Downstream tasks | Practical value? |
| Efficiency | NFE, Training time | Computational cost? |

#### 3. Ablation Studies

Systematically vary one component at a time:
- Architecture (U-Net vs. Transformer)
- Noise schedule (linear vs. cosine)
- Training procedure (batch size, learning rate)
- Sampling method (DDPM vs. DDIM)

**Example ablation**:
```
Baseline: FID = 15.2
+ Cosine schedule: FID = 12.8 (improvement)
+ Larger batch: FID = 11.5 (improvement)
+ EMA: FID = 10.9 (improvement)
```

#### 4. Statistical Significance

**Best practices**:

- Multiple random seeds (at least 3-5)
- Report mean ± standard deviation
- Statistical tests (t-test, bootstrap)
- Confidence intervals

**Example**:
```
Model A: FID = 12.3 ± 0.5
Model B: FID = 11.8 ± 0.6
Difference: 0.5 (not significant, p=0.15)
```

#### 5. Computational Budget

**Report full costs**:

- Training: GPU-hours, wall-clock time
- Sampling: Seconds per sample, NFE
- Memory: Peak GPU memory
- Total cost: Dollar cost estimate

**Example comparison**:
```
Model A: FID=10, 100 GPU-hours training, 1s/sample
Model B: FID=12, 10 GPU-hours training, 0.1s/sample
→ Model B may be better for deployment
```

---

## Evaluation for Biological Data

### Unique Challenges

1. **High dimensionality**: Thousands of genes
2. **Sparse structure**: Many zeros (scRNA-seq)
3. **Biological constraints**: Pathway structure, regulatory networks
4. **Limited ground truth**: No "correct" cell state
5. **Batch effects**: Technical variability

### Recommended Metrics

#### Statistical Fidelity
- **Marginal distributions**: Per-gene statistics
- **Correlation structure**: Gene-gene correlations
- **Higher-order moments**: Skewness, kurtosis

#### Biological Validity
- **Pathway enrichment**: Gene set enrichment analysis (GSEA)
- **Cell type markers**: Known marker gene expression
- **Regulatory networks**: Transcription factor activity
- **Developmental trajectories**: Pseudotime consistency

#### Downstream Utility
- **Cell type classification**: Train classifier on synthetic, test on real
- **Perturbation prediction**: Drug response accuracy
- **Batch correction**: Integration with real data
- **Data augmentation**: Improved downstream task performance

#### Epiplexity-Based Evaluation (NEW)

**Setup**:
1. Train observer model on real data
2. Train observer model on synthetic data
3. Compare loss trajectories

**Metrics**:

$$
\text{Epiplexity proxy} = \int_0^T (L(t) - L_\infty) dt
$$

**Interpretation**:

- High epiplexity → synthetic data teaches biological structure
- Low epiplexity → synthetic data is noise-matching without structure

**Advantages**:

- Detects whether synthetic data contains learnable biology
- Goes beyond statistical matching
- Measures utility for downstream learning

---

## Practical Evaluation Workflow

### Step 1: Baseline Metrics
Compute standard metrics for quick assessment:
- FID (if applicable)
- Likelihood/ELBO
- Visual inspection (if images)

### Step 2: Comprehensive Evaluation
If baseline looks promising:
- Precision/Recall
- Multiple sample sizes
- Ablation studies
- Statistical significance tests

### Step 3: Domain-Specific Validation
For your specific application:
- Downstream task performance
- Expert evaluation
- Biological validation (for bio data)
- Epiplexity analysis

### Step 4: Efficiency Analysis
Practical deployment considerations:
- Training cost
- Sampling speed
- Memory requirements
- Scalability

### Step 5: Reporting
**Minimum reporting standards**:

- Multiple metrics (quality, diversity, efficiency)
- Statistical significance (multiple seeds)
- Computational costs
- Failure cases and limitations
- Reproducibility information (code, hyperparameters)

---

## Common Pitfalls

### 1. Single-Metric Optimization
**Problem**: Optimizing only FID can hurt diversity
**Solution**: Use multiple complementary metrics

### 2. Cherry-Picking Samples
**Problem**: Showing only best samples
**Solution**: Report aggregate statistics, show random samples

### 3. Ignoring Computational Cost
**Problem**: Achieving SOTA quality with 10x compute
**Solution**: Report efficiency metrics, Pareto frontiers

### 4. Overfitting to Validation Set
**Problem**: Tuning hyperparameters on test set
**Solution**: Proper train/val/test splits, hold-out evaluation

### 5. Ignoring Statistical Significance
**Problem**: Claiming improvement from noise
**Solution**: Multiple seeds, significance tests

### 6. Domain Mismatch
**Problem**: Using ImageNet-trained metrics for non-image data
**Solution**: Domain-appropriate metrics, custom evaluations

---

## Decision Framework

### When is Model A "Better" than Model B?

**Model A is strictly better if**:

- Better on all metrics
- Same computational cost
- No significant trade-offs

**Model A is better for research if**:

- Significantly better on primary metric (e.g., FID)
- Novel capabilities or insights
- Reasonable computational cost

**Model A is better for deployment if**:

- Sufficient quality for application
- Much more efficient
- More robust and reliable

**Model A is better for your specific use case if**:

- Better on task-specific metrics
- Better downstream performance
- Fits your computational budget

---

## Summary

### Key Principles

1. **No single metric**: Use multiple complementary metrics
2. **Define objectives**: Research vs. application goals differ
3. **Statistical rigor**: Multiple seeds, significance tests
4. **Report costs**: Computational efficiency matters
5. **Domain-specific**: Adapt metrics to your data type
6. **Utility matters**: Downstream task performance is crucial

### Recommended Metric Suites

**For image diffusion models**:

- FID (primary)
- Precision/Recall (quality-diversity)
- Sampling efficiency (NFE, time)
- Human evaluation (if budget allows)

**For biological data**:

- Statistical fidelity (correlations, distributions)
- Biological validity (pathways, markers)
- Downstream utility (classification, prediction)
- Epiplexity (learnable structure)

**For general generative models**:

- Quality metric (FID, IS, or domain-specific)
- Diversity metric (Recall, mode coverage)
- Likelihood (if tractable)
- Efficiency (training and sampling cost)
- Task performance (downstream applications)

### The Bottom Line

> **A model is "better" when it achieves your specific objectives more effectively, considering quality, diversity, efficiency, and utility—validated through rigorous multi-metric evaluation.**

---

## Related Documents

- [Epiplexity: From Entropy to Epiplexity](epiplexity/01_from_entropy_to_epiplexity.md) — Observer-dependent information
- [Why Random Noise Has High Entropy but Low Epiplexity](epiplexity/02_random_noise.md) — Structure vs. randomness
- [DDPM Foundations](../DDPM/01_ddpm_foundations.md) — Diffusion model theory
- [DDPM Training](../DDPM/02_ddpm_training.md) — Training considerations
- [DDPM Sampling](../DDPM/03_ddpm_sampling.md) — Sampling methods and efficiency

---

## References

### Evaluation Metrics
1. **Salimans, T., et al. (2016)**. Improved Techniques for Training GANs. *NeurIPS*. (Inception Score)
2. **Heusel, M., et al. (2017)**. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *NeurIPS*. (FID)
3. **Kynkäänniemi, T., et al. (2019)**. Improved Precision and Recall Metric for Assessing Generative Models. *NeurIPS*.
4. **Zhang, R., et al. (2018)**. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*. (LPIPS)

### Diffusion Model Evaluation
5. **Ho, J., et al. (2020)**. Denoising Diffusion Probabilistic Models. *NeurIPS*.
6. **Song, J., et al. (2021)**. Denoising Diffusion Implicit Models. *ICLR*.
7. **Dhariwal, P., & Nichol, A. (2021)**. Diffusion Models Beat GANs on Image Synthesis. *NeurIPS*.

### Epiplexity
8. **From Entropy to Epiplexity** (2024). [arXiv:2601.03220](https://arxiv.org/abs/2601.03220)

### Best Practices
9. **Borji, A. (2019)**. Pros and Cons of GAN Evaluation Measures. *Computer Vision and Image Understanding*.
10. **Theis, L., et al. (2016)**. A Note on the Evaluation of Generative Models. *ICLR*.
