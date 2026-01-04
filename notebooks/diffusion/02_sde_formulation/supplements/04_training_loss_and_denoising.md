# Understanding the Training Loss: How Learning to Predict Score = Learning to Denoise

## The Loss Function

$$
\mathcal{L} = \|s_\theta(x_t, t) - (-\varepsilon/\sigma_t)\|^2
$$

### What Each Term Means

- **$s_\theta(x_t, t)$**: The neural network's prediction of the score at noisy state $x_t$ and time $t$
- **$-\varepsilon/\sigma_t$**: The **true target score** (what we want the network to predict)
- **$\varepsilon$**: The noise that was added to create $x_t$ from $x_0$
- **$\sigma_t = \sqrt{1-\bar{\alpha}_t}$**: The noise standard deviation at time $t$

### What the Loss Measures

The loss measures: **How well does the network predict the true score?**

When the loss is small, $s_\theta(x_t, t) \approx -\varepsilon/\sigma_t$, meaning the network has learned to identify:
- Which direction points toward higher probability (the score)
- Which is equivalent to identifying the noise that was added

---

## Why $-\varepsilon/\sigma_t$ Is the Target Score

### Step 1: The Forward Process

We corrupt clean data $x_0$ into noisy data $x_t$:

$$
x_t = \alpha_t x_0 + \sigma_t \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

where $\alpha_t = \sqrt{\bar{\alpha}_t}$ and $\sigma_t = \sqrt{1-\bar{\alpha}_t}$.

### Step 2: The Conditional Distribution

Given $x_0$, the noisy state $x_t$ follows:

$$
p_t(x_t \mid x_0) = \mathcal{N}(x_t; \alpha_t x_0, \sigma_t^2 I)
$$

### Step 3: Computing the Score

For a Gaussian $\mathcal{N}(\mu, \Sigma)$, the score is:

$$
\nabla_x \log p(x) = -\Sigma^{-1}(x - \mu)
$$

Applying this:

$$
\nabla_x \log p_t(x_t \mid x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2}
$$

### Step 4: Expressing in Terms of Noise

From the forward process: $x_t - \alpha_t x_0 = \sigma_t \varepsilon$

Substitute:

$$
\nabla_x \log p_t(x_t \mid x_0) = -\frac{\sigma_t \varepsilon}{\sigma_t^2} = \boxed{-\frac{\varepsilon}{\sigma_t}}
$$

**This is why the target is $-\varepsilon/\sigma_t$**: It's the analytical score for the conditional distribution $p_t(x_t \mid x_0)$.

---

## Why Minimizing This Loss Teaches Denoising

### The Key Insight: Score = Denoising Direction

The score function $\nabla_x \log p_t(x)$ points in the direction of **steepest increase in log probability**. In the context of diffusion:

- **Higher probability regions** = regions with more data-like structure
- **Lower probability regions** = regions with more noise
- **Following the score** = moving from noise toward data = **denoising**

### The Training Process

1. **We know the noise** $\varepsilon$ (we added it!)
2. **We compute the true score** $-\varepsilon/\sigma_t$ (analytically)
3. **We train the network** to predict this score
4. **The network learns** to identify the denoising direction

### What Happens During Training

At each iteration:

```
1. Start with clean data: x_0
2. Add noise: x_t = α_t x_0 + σ_t ε
3. Network sees: (x_t, t)
4. Network predicts: s_θ(x_t, t) ≈ -ε/σ_t
5. Loss measures: ||s_θ(x_t, t) - (-ε/σ_t)||²
6. Backprop updates θ to reduce loss
```

**After many iterations**, the network learns:
- Given any noisy $x_t$ at any time $t$
- Predict the score $s_\theta(x_t, t)$ that points toward cleaner data

### Why This Works: The Reverse SDE Connection

During **generation**, we solve the reverse SDE:

$$
dx = \left[f(x,t) - g(t)^2 s_\theta(x,t)\right]dt + g(t)\,dw
$$

**Important reminder**: $f(x,t)$ and $g(t)$ are **design choices** (not learned). See [`01_forward_sde_design_choices.md`](./01_forward_sde_design_choices.md) for details on how to choose them.

The term $-g(t)^2 s_\theta(x,t)$ is the **denoising force**:
- $s_\theta(x,t)$ points toward higher probability (less noise) — **This is learned**
- $g(t)^2$ scales it appropriately — **This is fixed** (from your forward SDE choice)
- This term pulls the sample from noise toward data

**The network learned to denoise during training** because:
- Training: Learn to predict the score (which points toward $x_0$)
- Generation: Use the score in the reverse SDE to actually denoise

**Key insight**: The reverse SDE inherits $f(x,t)$ and $g(t)$ from your forward SDE design. You only train the score $s_\theta(x,t)$.

---

## The Learning Objective: Denoising Score Matching

The loss function implements **denoising score matching** (Vincent, 2011):

### The General Principle

Instead of learning the score of the marginal $p_t(x)$ (hard—we don't have samples), we learn the score of the **conditional** $p_t(x \mid x_0)$ (easy—we know $x_0$).

### Why This Is Equivalent

Under mild conditions, learning the conditional score at all $(x_0, t)$ pairs is equivalent to learning the marginal score:

$$
\mathbb{E}_{x_0 \sim p_{\text{data}}} \left[\nabla_x \log p_t(x_t \mid x_0)\right] = \nabla_x \log p_t(x_t)
$$

**Intuition**: The marginal score is the average of conditional scores over all possible $x_0$.

### The Training Objective

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \varepsilon} \left[\lambda(t) \left\| s_\theta(x_t, t) - \nabla_x \log p_t(x_t \mid x_0) \right\|^2\right]
$$

where:
- $t \sim \text{Uniform}(0, T)$: Random timestep
- $x_0 \sim p_{\text{data}}$: Clean data sample
- $\varepsilon \sim \mathcal{N}(0, I)$: Random noise
- $x_t = \alpha_t x_0 + \sigma_t \varepsilon$: Noisy data
- $\lambda(t)$: Weighting function (often $\lambda(t) = \sigma_t^2$)

### Why Weight by $\lambda(t) = \sigma_t^2$?

The weighting compensates for the fact that:
- At high noise ($\sigma_t$ large), the score magnitude is smaller ($\propto 1/\sigma_t$)
- Without weighting, the loss would be dominated by low-noise timesteps
- Weighting by $\sigma_t^2$ balances learning across all noise levels

---

## The Complete Picture: Training → Generation

### Training Phase

```
Goal: Learn s_θ(x_t, t) ≈ ∇_x log p_t(x_t)

Method:
1. Sample (x_0, t, ε)
2. Create x_t = α_t x_0 + σ_t ε
3. Compute target: -ε/σ_t
4. Predict: s_θ(x_t, t)
5. Minimize: ||s_θ(x_t, t) - (-ε/σ_t)||²
```

**Result**: Network learns to identify denoising direction at all noise levels.

### Generation Phase

```
Goal: Generate x_0 from noise x_T

Method:
1. Start: x_T ~ N(0, I)
2. Solve reverse SDE:
   dx = [f(x,t) - g(t)² s_θ(x,t)] dt + g(t) dw
3. The term -g(t)² s_θ(x,t) denoises by following the score
4. End: x_0 (generated sample)
```

**Result**: Network's learned score guides the denoising process.

---

## Intuitive Analogy

Think of training a **compass**:

1. **Training**: 
   - You're in a foggy forest (noisy $x_t$)
   - You know where you started ($x_0$)
   - You know which way is "home" (the score $-\varepsilon/\sigma_t$)
   - You train a compass (network) to point home

2. **Generation**:
   - You're lost in fog (pure noise $x_T$)
   - You use the compass (learned score $s_\theta$)
   - You follow it step by step (solve reverse SDE)
   - You reach home (clean data $x_0$)

The loss function is teaching the compass to always point toward home, regardless of where you are in the fog.

---

## Why This Is Better Than Direct Denoising

You might wonder: "Why not just train a network to predict $x_0$ from $x_t$ directly?"

### Problems with Direct Prediction

1. **Mode collapse**: The network might average over multiple possible $x_0$ values
2. **Blurry outputs**: Averaging creates smooth but unrealistic images
3. **No diversity**: Deterministic mapping $x_t \to x_0$ gives same output every time

### Advantages of Score Matching

1. **Learns the gradient field**: Captures the structure of the data manifold
2. **Stochastic generation**: The reverse SDE adds noise, creating diverse samples
3. **Better mode coverage**: The score field guides toward all modes, not just averages

---

## Summary

| Question | Answer |
|----------|--------|
| **What does the loss measure?** | How well the network predicts the true score |
| **Why is the target $-\varepsilon/\sigma_t$?** | It's the analytical score of $p_t(x_t \mid x_0)$ |
| **How does this teach denoising?** | The score points toward data; learning the score = learning denoising direction |
| **Why backpropagation works?** | Minimizing loss makes $s_\theta$ approximate the true score at all $(x_t, t)$ |
| **How is this used in generation?** | The learned score guides the reverse SDE to denoise from $x_T$ to $x_0$ |

**The magic**: By learning to predict the score (which we can compute analytically during training), the network implicitly learns to denoise, even though it never sees a denoising task during training!

---

## References

- **Vincent (2011)**: "A Connection Between Score Matching and Denoising Autoencoders" — Original denoising score matching
- **Song & Ermon (2019)**: "Generative Modeling by Estimating Gradients of the Data Distribution" — Score-based generative models
- **Ho et al. (2020)**: "Denoising Diffusion Probabilistic Models" — DDPM (equivalent to score matching)
- **Song et al. (2021)**: "Score-Based Generative Modeling through SDEs" — SDE formulation

