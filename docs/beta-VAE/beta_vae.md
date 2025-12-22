# β-VAE: Disentanglement and the Information Bottleneck

Building on the VAE foundation, β-VAE introduces a single hyperparameter that controls the trade-off between reconstruction quality and latent space structure.

---

## 1. Motivation: Why Modify the VAE?

Standard VAEs optimize:

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \mathrm{KL}(q(z|x) \| p(z))
$$

This treats reconstruction and regularization equally. But what if we want:

- **More structured latent space** → increase KL weight
- **Better reconstruction** → decrease KL weight

β-VAE makes this explicit.

---

## 2. The β-VAE Objective

Simply multiply the KL term by β:

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta \cdot \mathrm{KL}(q(z|x) \| p(z))
$$

| β value | Effect |
|---------|--------|
| β = 1 | Standard VAE |
| β > 1 | Stronger regularization → more disentanglement, worse reconstruction |
| β < 1 | Weaker regularization → better reconstruction, less structure |

---

## 3. Information Bottleneck Interpretation

β-VAE can be understood through the **information bottleneck** framework:

$$
\max_{q(z|x)} \; I(z; y) - \beta \cdot I(z; x)
$$

Where:
- $I(z; y)$ — mutual information between latent and target (reconstruction)
- $I(z; x)$ — mutual information between latent and input (compression)

Higher β forces the model to:
1. Compress more aggressively
2. Keep only the most informative features
3. Discard nuisance variation

---

## 4. Disentanglement: What Does It Mean?

A **disentangled representation** has:

- Each latent dimension captures **one independent factor of variation**
- Changing one dimension changes one semantic attribute
- Dimensions are statistically independent

### Example (Images)

| Dimension | Controls |
|-----------|----------|
| $z_1$ | Rotation |
| $z_2$ | Scale |
| $z_3$ | Color |

### Example (Gene Expression)

| Dimension | Controls |
|-----------|----------|
| $z_1$ | Cell type identity |
| $z_2$ | Cell cycle phase |
| $z_3$ | Batch effect |

---

## 5. Why β > 1 Encourages Disentanglement

The KL term can be decomposed:

$$
\mathrm{KL}(q(z|x) \| p(z)) = \underbrace{I(z; x)}_{\text{mutual info}} + \underbrace{\mathrm{KL}(q(z) \| p(z))}_{\text{marginal matching}}
$$

Increasing β:
1. **Reduces $I(z; x)$** — forces compression
2. **Pushes $q(z)$ toward $p(z) = \mathcal{N}(0, I)$** — encourages independence

The factorial prior $p(z) = \prod_i p(z_i)$ induces statistical independence between dimensions.

---

## 6. The Reconstruction-Disentanglement Trade-off

This is the fundamental tension:

```
β small ←————————————————————→ β large
Better reconstruction          Better disentanglement
Entangled latents              Worse reconstruction
Overfitting risk               Posterior collapse risk
```

### Posterior Collapse

When β is too high:
- $q(z|x) \approx p(z)$ for all $x$
- Latent carries no information
- Decoder ignores $z$, generates "average" output

---

## 7. Disentanglement Metrics

### DCI (Disentanglement, Completeness, Informativeness)

- **Disentanglement**: Does each code capture at most one factor?
- **Completeness**: Is each factor captured by at most one code?
- **Informativeness**: Can factors be predicted from codes?

### MIG (Mutual Information Gap)

$$
\text{MIG} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{H(v_k)} \left( I(z_{j^{(k)}}; v_k) - \max_{j \neq j^{(k)}} I(z_j; v_k) \right)
$$

Measures the gap between the most and second-most informative latent for each factor.

### SAP (Separated Attribute Predictability)

Trains classifiers to predict factors from individual latents.

---

## 8. Implementation

### Loss Function

```python
def beta_vae_loss(x, x_recon, mu, logvar, beta=4.0):
    """β-VAE loss with configurable β."""
    # Reconstruction (negative log-likelihood)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # β-weighted ELBO
    return recon_loss + beta * kl_loss
```

### Annealing Strategies

Rather than fixed β, gradually increase it:

```python
def get_beta(epoch, warmup_epochs=10, target_beta=4.0):
    """Linear β annealing."""
    if epoch < warmup_epochs:
        return target_beta * epoch / warmup_epochs
    return target_beta
```

This helps avoid posterior collapse early in training.

---

## 9. Variants and Extensions

### β-TCVAE (Total Correlation VAE)

Decomposes KL into three terms:

$$
\mathrm{KL}(q(z|x) \| p(z)) = \underbrace{I(z; x)}_{\text{index-code MI}} + \underbrace{\mathrm{TC}(z)}_{\text{total correlation}} + \underbrace{\sum_i \mathrm{KL}(q(z_i) \| p(z_i))}_{\text{dimension-wise KL}}
$$

Only penalizes the **total correlation** term, which directly measures dependence between dimensions.

### Factor VAE

Adds an adversarial term to encourage factorial $q(z)$:

$$
\mathcal{L} = \mathcal{L}_{\text{VAE}} + \gamma \cdot \mathrm{KL}(q(z) \| \bar{q}(z))
$$

Where $\bar{q}(z) = \prod_i q(z_i)$ is the factorial approximation.

### DIP-VAE (Disentangled Inferred Prior)

Matches moments of $q(z)$ to the prior:

$$
\mathcal{L} = \mathcal{L}_{\text{VAE}} + \lambda_{\text{od}} \sum_{i \neq j} [\text{Cov}(z)]_{ij}^2 + \lambda_d \sum_i ([\text{Cov}(z)]_{ii} - 1)^2
$$

---

## 10. Application to Gene Expression

### Why Disentanglement Matters

In single-cell biology, we want latents that separate:
- **Biological signal** (cell type, state)
- **Technical noise** (batch, sequencing depth)

A disentangled model enables:
1. **Batch correction**: Zero out batch dimensions
2. **Counterfactuals**: Change disease dimension, keep cell type
3. **Interpretability**: Each dimension has biological meaning

### Practical Considerations

| Challenge | Solution |
|-----------|----------|
| No ground truth factors | Use known covariates (batch, donor) as proxies |
| High dimensionality | Start with PCA-reduced input |
| Sparse data | Use negative binomial likelihood |

---

## 11. Experiments to Run

### Experiment 1: β Sweep

```python
betas = [0.1, 0.5, 1.0, 2.0, 4.0, 10.0]
for beta in betas:
    model = BetaVAE(beta=beta)
    train(model)
    evaluate_reconstruction(model)
    evaluate_disentanglement(model)
```

### Experiment 2: Latent Traversal

For each dimension $i$:
1. Encode a sample: $\mu, \sigma = \text{encode}(x)$
2. Vary $z_i$ from $-3$ to $+3$
3. Decode and visualize

### Experiment 3: Condition Prediction

Train linear classifiers to predict tissue/disease from individual latent dimensions.

---

## 12. Connection to Diffusion

β-VAE's insight—that **compression induces structure**—reappears in diffusion:

- Diffusion adds noise (compression) then learns to denoise
- The noise schedule is analogous to β annealing
- Both trade reconstruction for latent regularity

This is why understanding β-VAE deeply prepares you for diffusion.

---

## 13. References

1. **Higgins et al.** (2017) — "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
2. **Chen et al.** (2018) — "Isolating Sources of Disentanglement in VAEs" (β-TCVAE)
3. **Kim & Mnih** (2018) — "Disentangling by Factorising" (Factor VAE)
4. **Kumar et al.** (2018) — "Variational Inference of Disentangled Latent Concepts from Unlabeled Observations" (DIP-VAE)
5. **Locatello et al.** (2019) — "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"

---

## Next Steps

After β-VAE:
1. **IWAE** — Tighter bounds without changing the objective structure
2. **Score matching** — The bridge to diffusion models

See [ROADMAP.md](../ROADMAP.md) for the full learning path.
