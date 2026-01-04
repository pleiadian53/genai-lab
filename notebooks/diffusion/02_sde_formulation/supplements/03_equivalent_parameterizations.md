# Equivalence of Score, Noise, and Clean Data Parameterizations

## The Three Parameterizations

A neural network in a diffusion model can predict any of these:

1. **Score**: $s_\theta(x_t, t) \approx \nabla_x \log p_t(x_t)$
2. **Noise**: $\varepsilon_\theta(x_t, t) \approx \varepsilon$
3. **Clean data**: $\hat{x}_0(x_t, t) \approx x_0$

**Claim**: These are mathematically equivalent—you can convert between them.

---

## The Forward Process (Starting Point)

The forward process corrupts clean data $x_0$ into noisy data $x_t$:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

where:
- $\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)$ (cumulative signal retention)
- $\sqrt{\bar{\alpha}_t}$ scales the signal
- $\sqrt{1-\bar{\alpha}_t}$ scales the noise

This is the **closed-form marginal** of the VP-SDE.

---

## Deriving the Conditional Score

The conditional distribution is:

$$
p_t(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t)\, I\right)
$$

For a Gaussian $\mathcal{N}(\mu, \Sigma)$, the score is:

$$
\nabla_x \log p(x) = -\Sigma^{-1}(x - \mu)
$$

Applying this:

$$
\nabla_x \log p_t(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0}{1 - \bar{\alpha}_t}
$$

---

## The Key Relationship

From the forward process:
$$
x_t - \sqrt{\bar{\alpha}_t}\, x_0 = \sqrt{1-\bar{\alpha}_t}\, \varepsilon
$$

Substitute into the score:

$$
\nabla_x \log p_t(x_t \mid x_0) = -\frac{\sqrt{1-\bar{\alpha}_t}\, \varepsilon}{1 - \bar{\alpha}_t} = \boxed{-\frac{\varepsilon}{\sqrt{1-\bar{\alpha}_t}}}
$$

**This is the fundamental relationship**: Score = scaled noise.

---

## Conversion Formulas

Let $\sigma_t = \sqrt{1-\bar{\alpha}_t}$ (noise standard deviation) and $\alpha_t = \sqrt{\bar{\alpha}_t}$ (signal scale).

### Score ↔ Noise

$$
\boxed{s_\theta(x_t, t) = -\frac{\varepsilon_\theta(x_t, t)}{\sigma_t}}
$$

$$
\boxed{\varepsilon_\theta(x_t, t) = -\sigma_t \cdot s_\theta(x_t, t)}
$$

### Noise ↔ Clean Data

From $x_t = \alpha_t x_0 + \sigma_t \varepsilon$, solve for $x_0$:

$$
\boxed{\hat{x}_0 = \frac{x_t - \sigma_t \varepsilon_\theta(x_t, t)}{\alpha_t}}
$$

$$
\boxed{\varepsilon_\theta(x_t, t) = \frac{x_t - \alpha_t \hat{x}_0}{\sigma_t}}
$$

### Score ↔ Clean Data

Combine the above:

$$
\boxed{\hat{x}_0 = \frac{x_t + \sigma_t^2 s_\theta(x_t, t)}{\alpha_t}}
$$

$$
\boxed{s_\theta(x_t, t) = \frac{\alpha_t \hat{x}_0 - x_t}{\sigma_t^2}}
$$

---

## Summary Table

| If you have... | To get Score | To get Noise | To get Clean Data |
|----------------|--------------|--------------|-------------------|
| **Score** $s$ | — | $\varepsilon = -\sigma_t s$ | $\hat{x}_0 = \frac{x_t + \sigma_t^2 s}{\alpha_t}$ |
| **Noise** $\varepsilon$ | $s = -\varepsilon/\sigma_t$ | — | $\hat{x}_0 = \frac{x_t - \sigma_t \varepsilon}{\alpha_t}$ |
| **Clean Data** $\hat{x}_0$ | $s = \frac{\alpha_t \hat{x}_0 - x_t}{\sigma_t^2}$ | $\varepsilon = \frac{x_t - \alpha_t \hat{x}_0}{\sigma_t}$ | — |

Where: $\alpha_t = \sqrt{\bar{\alpha}_t}$, $\sigma_t = \sqrt{1-\bar{\alpha}_t}$

---

## Why Different Frameworks Use Different Parameterizations

### DDPM (Ho et al. 2020): Predicts Noise $\varepsilon$

**Reason**: Empirically more stable training. The noise $\varepsilon \sim \mathcal{N}(0, I)$ has a consistent scale across all timesteps, whereas the score magnitude varies with $t$.

**Loss**:
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]
$$

### Score-Based Models (Song et al. 2019): Predicts Score $s$

**Reason**: Directly motivated by score matching theory. The score has a clear interpretation as the gradient of log-density.

**Loss** (denoising score matching):
$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, x_t}\left[\|s_\theta(x_t, t) - \nabla_x \log p_t(x_t \mid x_0)\|^2\right]
$$

### v-prediction (Salimans & Ho 2022): Predicts a Combination

**Why?** For some noise schedules, predicting a linear combination of noise and data works better:
$$
v_t = \alpha_t \varepsilon - \sigma_t x_0
$$

This balances the learning signal across timesteps.

---

## Intuitive Understanding

All three quantities answer the same question from different angles:

| Parameterization | Question Answered |
|------------------|-------------------|
| **Score** $\nabla_x \log p_t(x)$ | "Which direction increases probability?" |
| **Noise** $\varepsilon$ | "What random noise was added to the clean data?" |
| **Clean data** $x_0$ | "What was the original data before corruption?" |

Given the forward process, knowing any one of these determines the other two.

---

## Practical Example

Suppose at timestep $t$:
- $\bar{\alpha}_t = 0.5$ (so $\alpha_t = \sqrt{0.5} \approx 0.707$, $\sigma_t = \sqrt{0.5} \approx 0.707$)
- Current noisy state: $x_t = [1.0, 2.0]$
- Network predicts noise: $\varepsilon_\theta = [0.5, 1.0]$

Then:
- **Score**: $s = -\varepsilon/\sigma_t = -[0.5, 1.0]/0.707 \approx [-0.707, -1.414]$
- **Clean data**: $\hat{x}_0 = (x_t - \sigma_t \varepsilon)/\alpha_t = ([1.0, 2.0] - 0.707 \cdot [0.5, 1.0])/0.707 \approx [0.914, 1.828]$

All three representations contain the same information, just expressed differently.

---

## Key Takeaway

**The forward process equation**:
$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon
$$

is the Rosetta Stone that connects all three parameterizations. Given any two of $(x_t, x_0, \varepsilon)$ and the noise schedule $(\alpha_t, \sigma_t)$, you can compute the third—and from there, derive the score.

---

## References

- **Ho et al. (2020)**: DDPM — Uses noise prediction
- **Song & Ermon (2019)**: Score-based models — Uses score prediction
- **Salimans & Ho (2022)**: Progressive Distillation — Introduces v-prediction
- **Karras et al. (2022)**: EDM — Analyzes different parameterizations systematically

