# Using VAEs for Prediction: From Generative Models to Downstream Tasks

VAEs, cVAEs, and β-VAEs are generative models—they learn to reconstruct data, not predict labels. Yet they are remarkably effective for prediction tasks when used correctly.

This document explains how to leverage VAE-family models for classification, regression, ranking, and anomaly detection in computational biology applications.

---

## The Core Insight

VAEs don't predict labels by default, but they *manufacture representations* that can be turned into powerful predictors. The key is understanding what the latent space captures and how to extract predictions from it.

**The pattern:**

1. Train a VAE to model gene expression
2. Extract latent representations $z$ for each sample
3. Use $z$ as features for downstream prediction tasks

This two-stage approach often outperforms end-to-end discriminative models because the VAE learns robust, denoised representations of the underlying biology.

---

## 1. What a VAE Optimizes (and What It Doesn't)

A vanilla VAE learns:

- An **encoder** $q_\phi(z | x)$: gene expression → latent state
- A **decoder** $p_\theta(x | z)$: latent state → reconstructed expression

The objective is the ELBO:

- Reconstruct $x$ well
- Keep $z$ close to a simple prior (usually $\mathcal{N}(0, I)$)

No labels. No prediction target. Just *"explain the data compactly."*

By default, a VAE is:

- Not a classifier
- Not a regressor
- Not a ranker

But that's like saying PCA can't predict anything. True—and also missing the point.

---

## 2. Prediction Lives in Latent Space

Once you have a latent variable $z$, three prediction strategies become available.

### Strategy A: Latent → Downstream Predictor

This is the most common and most powerful approach:

```
gene expression x
   ↓ encoder
latent z
   ↓ classifier / regressor / ranker
prediction ŷ
```

This is not a hack—it's the *intended use* in biology.

**Examples:**

- scRNA-seq: latent → cell type classification
- Bulk RNA-seq: latent → disease state prediction
- Transcript modeling: latent → reliability score

The VAE performs **representation learning**, not prediction. The predictor is a separate model trained on $z$.

**Why this works well:**

- Expression data is noisy; VAEs denoise and compress
- Reliability is latent and indirect; $z$ captures the underlying state
- The latent space is smooth and continuous

You're essentially asking: *"What is the biological state of this sample?"* rather than *"What are the raw counts?"*

---

## 3. cVAE: Controlled Inference via Conditioning

A conditional VAE models:

$$
p(x | z, c)
$$

where $c$ might be:

- Tissue type
- Disease state
- Batch ID
- Perturbation condition

Now the latent $z$ is forced to explain *what remains* after conditioning. For gene expression, this is powerful.

### Why cVAE Excels for Downstream Prediction

You can:

- Condition on tissue and disease
- Force $z$ to capture sample-specific variation
- Decouple biological signal from confounders

```python
z = encoder(x, condition)
y_pred = prediction_head(z)
```

The prediction now answers: *"What is the state of this sample given its biological context?"*

---

## 4. β-VAE: Interpretability Through Disentanglement

β-VAE modifies the ELBO:

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}[\log p(x|z)] - \beta \cdot \text{KL}(q(z|x) \| p(z))
$$

Increasing $\beta$:

- Enforces factorized latent dimensions
- Sacrifices reconstruction fidelity
- Gains **disentanglement**

For biology, this often means:

- One latent dimension ≈ expression strength
- Another ≈ variability across conditions
- Another ≈ batch susceptibility

Now prediction becomes not just possible, but *explainable*:

> "This sample is unusual because latent factor 3 (expression instability) is high."

This pairs well with:

- Ranking samples by specific latent factors
- Thresholding on interpretable dimensions
- Model interpretation and feature attribution

---

## 5. Joint Models: Prediction Inside the VAE

You can also bake prediction directly into the VAE.

### Approach 1: VAE + Supervised Head

Loss:

$$
\mathcal{L} = \mathcal{L}_{\text{ELBO}}(x) + \lambda \cdot \mathcal{L}_{\text{pred}}(y, f(z))
$$

This gives you:

- Generative modeling of expression
- Predictive latent space
- Semi-supervised learning (labels optional)

This is attractive when:

- Labels are sparse
- Unlabeled samples dominate
- Labels are noisy or partial

### Approach 2: Probabilistic Prediction

Instead of predicting a point estimate, model:

$$
p(y | z)
$$

Now the model outputs uncertainty:

> "This sample has a 0.78 ± 0.12 probability of belonging to class A."

This enables Bayesian decision-making and calibrated ranking.

---

## 6. Ranking and Anomaly Detection

VAEs are surprisingly effective for unsupervised ranking.

### Reconstruction Error as Anomaly Score

If a sample:

- Reconstructs poorly
- Has high posterior uncertainty
- Lives far from the latent manifold

...that's a signal worth investigating.

This makes VAEs naturally suited for:

- Outlier detection
- Quality screening
- Novelty detection

### Latent Likelihood as Confidence

You can rank samples by:

- ELBO (higher = more typical)
- Marginal likelihood
- Posterior entropy (lower = more confident)

This is ranking *without explicit labels*—purely from the generative model.

---

## 7. Practical Downstream Tasks for Evaluation

To evaluate generative models like VAEs, we need downstream prediction tasks. Here are practical options:

| Task | Labels | Difficulty | Biological Relevance |
|------|--------|------------|---------------------|
| **Cell type classification** | Cell type annotations | Easy | High |
| **Disease state prediction** | Case/control labels | Medium | High |
| **Batch prediction** | Batch IDs | Easy | Low (sanity check) |
| **Perturbation response** | Treatment labels | Hard | Very high |
| **Trajectory position** | Pseudotime | Medium | High |

### Multiple Complementary Approaches

1. **Latent → classifier**: Supervised prediction from $z$
2. **Latent uncertainty → quality proxy**: High variance = low confidence
3. **Reconstruction error → anomaly score**: Poorly modeled = unusual
4. **Condition-invariant factors**: What's consistent across contexts?
5. **Semi-supervised learning**: Leverage unlabeled data

A VAE is not competing with discriminative models—it's a *substrate* for building them.

---

## 8. Mental Model

Think of VAEs as answering:

> "What kind of thing is this sample?"

Not:

> "What is the label for this sample?"

But once you know what kind of thing it is, the second question becomes much easier—and much more robust.

---

## 9. Advanced Directions

Next steps for more sophisticated modeling:

- **Hierarchical VAEs**: Model structure at multiple scales (gene → pathway → program)
- **Mixture-of-VAEs**: Capture distinct regimes or subpopulations
- **Semi-supervised VAE**: Joint generative + discriminative training
- **Combining with other methods**: Use VAE latents as features for GMMs, PU learning, etc.

The generative story gives you *structure*. The predictive layer gives you *decisions*.

---

## Summary

| VAE Variant | Best For |
|-------------|----------|
| **VAE** | Unsupervised representation learning |
| **cVAE** | Controlled generation, removing confounders |
| **β-VAE** | Interpretable, disentangled representations |
| **Semi-supervised VAE** | Sparse labels, leveraging unlabeled data |

All can be used for downstream prediction by training classifiers/regressors on the latent space $z$.
