# Understanding Diffusion Models Through Stochastic Differential Equations

**A ground-up introduction to the SDE perspective on diffusion models**

This guide builds intuition for how diffusion models work by starting with stochastic differential equations (SDEs) as a mathematical framework, *before* connecting to DDPM or score matching. If you're coming from discrete-time DDPM, this will show you the continuous-time view that unifies all diffusion approaches.

---

## Table of Contents

1. [What is an SDE?](#1-what-is-an-sde)
2. [Understanding Each Symbol](#2-understanding-each-symbol)
3. [What is Chosen vs What is Learned](#3-what-is-chosen-vs-what-is-learned)
4. [Training Workflow Step-by-Step](#4-training-workflow-step-by-step)
5. [Sampling Workflow](#5-sampling-workflow)
6. [Connection to DDPM](#6-connection-to-ddpm)
7. [Concrete Example: VP-SDE](#7-concrete-example-vp-sde)

**Companion notebook**: [`02_sde_formulation.ipynb`](./02_sde_formulation.ipynb) implements these concepts with code and visualizations.

---

## 1. What is an SDE?

### Starting with ODEs

An **ordinary differential equation (ODE)** describes deterministic motion:

$$
\frac{dx(t)}{dt} = f(x(t), t)
$$

Given a starting point $x(0)$, the future trajectory is completely determined. Think of a ball rolling down a hill—if you know the initial position and velocity, you can predict exactly where it will be at any time.

**Example**: Exponential decay
$$

\frac{dx}{dt} = -\lambda x \quad \Rightarrow \quad x(t) = x(0) e^{-\lambda t}
$$

### Adding Randomness: SDEs

A **stochastic differential equation (SDE)** adds *continuous random noise* to this deterministic motion:

$$
dx(t) = f(x(t), t)\,dt + g(t)\,dw(t)
$$

This describes a **random process evolving in time**, not a single deterministic trajectory.

**Key difference:**

- **ODE**: One starting point → one path
- **SDE**: One starting point → distribution over paths

**Physical intuition**: A particle in a fluid experiences both:
- **Drift** $f(x,t)$: systematic force (gravity, electric field)
- **Diffusion** $g(t)\,dw(t)$: random collisions with molecules (Brownian motion)

---

## 2. Understanding Each Symbol

Let's break down the SDE equation term by term:

$$
dx(t) = \underbrace{f(x(t), t)}_{\text{drift}}\,dt + \underbrace{g(t)}_{\text{diffusion coefficient}}\,\underbrace{dw(t)}_{\text{Brownian motion}}
$$

### 2.1 State: $x(t)$

**What it is:**

- $x(t) \in \mathbb{R}^d$ is the system's state at time $t$
- In diffusion models: a noisy image, gene expression vector, or any data
- Dimension $d$ can be huge (millions for images)

**Crucial point**: $x(t)$ is a **random variable**, not a parameter. At each time $t$, there's a probability distribution $p_t(x)$ over possible states.

**Example**: For an image diffusion model:
- $t=0$: $x(0)$ is the clean image
- $t=0.5$: $x(0.5)$ is partially noisy
- $t=1$: $x(1)$ is pure noise

### 2.2 Time: $t$

**What it is:**

- Continuous time variable: $t \in [0, T]$
- Replaces discrete timesteps $t = 0, 1, 2, \ldots, T$ from DDPM

**Important**: This is **not physical time**. It's a continuous index for noise level:
- $t=0$: Clean data (no noise)
- $t=T$: Pure noise (data destroyed)

Think of it as a "corruption level" that smoothly varies from 0 to 100%.

### 2.3 Brownian Motion: $w(t)$

**What it is:**
The source of all randomness in the SDE. Also called a **Wiener process**.

**Mathematical properties:**
1. $w(0) = 0$ (starts at origin)
2. **Independent increments**: $w(t_2) - w(t_1)$ is independent of $w(t_1) - w(t_0)$
3. **Gaussian increments**: $w(t + \Delta t) - w(t) \sim \mathcal{N}(0, \Delta t)$
4. **Continuous but nowhere differentiable**: Infinitely jagged path

**Intuition**: Imagine a drunk person walking—each step is random, independent of previous steps, and the path gets more erratic over time.

**Key insight**: The differential $dw(t)$ behaves like:

$$
dw(t) \sim \sqrt{dt} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

This is why noise scales with $\sqrt{\text{time}}$, not linearly with time.

**Visualization**: If you plot $w(t)$ over $[0,1]$:
- It starts at 0
- Wanders randomly
- Typical displacement after time $t$ is $\mathcal{O}(\sqrt{t})$

### 2.4 Diffusion Coefficient: $g(t)$

**What it is:**
A scalar (or matrix) function that controls **how much randomness** is injected at time $t$.

**In the SDE:**
$$

dx = f(x,t)\,dt + g(t)\,dw(t)
$$

- If $g(t) = 0$: Pure ODE (deterministic)
- Larger $g(t)$: More noise added per unit time

**In diffusion models:**

- $g(t)$ is **chosen by you** (not learned)
- It defines the **noise schedule**
- Common choices:
  - Constant: $g(t) = \sigma$ (uniform noise)
  - Increasing: $g(t) = \beta(t)$ (more noise over time)
  - Variance-preserving: $g(t) = \sqrt{2\beta(t)}$

**Connection to DDPM**: The discrete noise schedule $\{\beta_t\}$ is the discretized version of $g(t)^2$.

### 2.5 Drift: $f(x, t)$

**What it is:**
The **deterministic component** of motion. It's the average direction and speed that $x(t)$ would move if noise were turned off.

**Mathematical definition:**
$$
f(x,t) = \mathbb{E}\left[\frac{dx}{dt} \,\Big|\, x(t) = x\right]
$$

The expected rate of change at state $x$ and time $t$.

**Physical intuition:**

- **Gravity**: Pulls objects down
- **Friction**: Slows motion proportional to velocity
- **Spring force**: Pulls toward equilibrium
- **Chemical gradient**: Drives diffusion toward lower concentration

**In diffusion models (forward process):**

Common choices:
1. **Zero drift**: $f(x,t) = 0$ (pure noise, VE-SDE)
2. **Linear drift**: $f(x,t) = -\frac{1}{2}\beta(t) x$ (variance-preserving, VP-SDE)

**Key point**: Like $g(t)$, the drift $f(x,t)$ is **chosen**, not learned.

---

## 3. What is Chosen vs What is Learned?

This is where most confusion arises. Let's be crystal clear.

### Forward SDE (Always Fixed)

**You choose:**

- Drift function: $f(x,t)$
- Diffusion coefficient: $g(t)$

These define a **known corruption process** that gradually destroys data structure.

$$
dx = f(x,t)\,dt + g(t)\,dw(t)
$$

**No learning happens here.** This is your design choice.

**Example choices:**

| SDE Type | Drift $f(x,t)$ | Diffusion $g(t)$ | Name |
|----------|----------------|------------------|------|
| VP-SDE | $-\frac{1}{2}\beta(t) x$ | $\sqrt{\beta(t)}$ | Variance-Preserving |
| VE-SDE | $0$ | $\sqrt{\frac{d\sigma^2(t)}{dt}}$ | Variance-Exploding |
| sub-VP | $-\frac{1}{2}\beta(t) x$ | $\sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})}$ | Sub-VP |

### Reverse SDE (Contains Learning)

The **reverse-time SDE** (going from noise back to data) is:

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\,d\bar{w}(t)
$$

where $\bar{w}(t)$ is a reverse-time Brownian motion.

**Breaking this down:**

- $f(x,t)$: Known (same as forward)
- $g(t)$: Known (same as forward)
- $\nabla_x \log p_t(x)$: **Unknown** — this is the **score function**

**The only thing we need to learn:**

$$
s_\theta(x, t) \approx \nabla_x \log p_t(x)
$$

A neural network $s_\theta$ that predicts the score (gradient of log-density) at any noise level $t$.

**Everything else is fixed mathematics.**

---

## 4. Training Workflow Step-by-Step

Here's exactly how you train a diffusion model in the SDE framework.

### Setup: What You Have Initially

1. **Dataset**: $\{x_0^{(i)}\}_{i=1}^N$ sampled from $p_{\text{data}}(x)$
2. **Chosen forward SDE**: $dx = f(x,t)\,dt + g(t)\,dw(t)$

This forward SDE defines:
- A family of distributions $\{p_t(x)\}_{t \in [0,T]}$
- One distribution per noise level

**Important**: You don't know $p_t(x)$ analytically (except at $t=0$ and $t=T$).

### Training Loop

**For each training iteration:**

#### Step 1: Sample Clean Data

$$
x_0 \sim \text{training dataset}
$$

Pick a random data point from your dataset.

#### Step 2: Sample a Timestep

$$
t \sim \text{Uniform}(0, T)
$$

Randomly choose a noise level. This ensures the network learns the score at **all** noise levels.

#### Step 3: Corrupt the Data (Simulate Forward SDE)

Generate noisy data by sampling from the conditional distribution:

$$
x_t \sim p_t(x \mid x_0)
$$

**How to do this in practice:**

For many SDEs (like VP-SDE), the marginal distribution $p_t(x \mid x_0)$ has a closed form:

$$
x_t = \alpha(t) x_0 + \sigma(t) \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

where $\alpha(t)$ and $\sigma(t)$ are determined by the SDE coefficients $f$ and $g$.

**Example (VP-SDE):**
$$

\alpha(t) = e^{-\frac{1}{2}\int_0^t \beta(s)ds}, \quad \sigma^2(t) = 1 - \alpha^2(t)
$$

#### Step 4: Compute Target Score

For Gaussian corruption, the conditional score is:

$$
\nabla_x \log p_t(x_t \mid x_0) = -\frac{x_t - \alpha(t) x_0}{\sigma^2(t)}
$$

This is **analytically available** because we chose the forward process.

**Connection to noise prediction:**
$$

\nabla_x \log p_t(x_t \mid x_0) = -\frac{\varepsilon}{\sigma(t)}
$$

where $\varepsilon$ is the noise we added. So predicting the score is equivalent to predicting the noise!

#### Step 5: Train Neural Network

Train a score network $s_\theta(x_t, t)$ to match the target score using **denoising score matching**:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, x_t} \left[\lambda(t) \left\| s_\theta(x_t, t) - \nabla_x \log p_t(x_t \mid x_0) \right\|^2\right]
$$

where $\lambda(t)$ is a weighting function (often $\lambda(t) = \sigma^2(t)$).

**In practice:**
```python
# Pseudocode
x_0 = sample_from_dataset()
t = random_uniform(0, T)
epsilon = random_normal(shape=x_0.shape)
x_t = alpha(t) * x_0 + sigma(t) * epsilon

# Predict score
score_pred = score_network(x_t, t)

# Target score
score_target = -epsilon / sigma(t)

# Loss
loss = mse_loss(score_pred, score_target)
```

**That's it.** This is the entire training procedure.

---

## 5. Sampling Workflow

After training, generate new samples by solving the reverse SDE.

### Step 1: Start from Noise

$$
x_T \sim \mathcal{N}(0, I)
$$

Sample pure Gaussian noise.

### Step 2: Solve Reverse SDE

Numerically integrate the reverse-time SDE from $t=T$ to $t=0$:

$$
dx = \left[f(x,t) - g(t)^2 s_\theta(x,t)\right] dt + g(t)\,d\bar{w}(t)
$$

**Discretization (Euler-Maruyama method):**

```python
x = sample_noise()
dt = -T / num_steps

for i in range(num_steps):
    t = T - i * dt
    
    # Drift term
    drift = f(x, t) - g(t)**2 * score_network(x, t)
    
    # Diffusion term
    diffusion = g(t) * random_normal(shape=x.shape)
    
    # Update
    x = x + drift * dt + diffusion * sqrt(-dt)

return x  # This is x_0 (generated sample)
```

### Alternative: Probability Flow ODE

You can also use the **deterministic** probability flow ODE (no stochasticity):

$$
\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)
$$

This generates samples via a deterministic path (like DDIM).

---

## 6. Connection to DDPM

How does this relate to the discrete-time DDPM you already know?

| DDPM Concept | SDE View |
|--------------|----------|
| Noise schedule $\{\beta_t\}$ | Diffusion coefficient $g(t)$ |
| Forward noising $q(x_t \mid x_{t-1})$ | Forward SDE |
| Reverse denoising $p_\theta(x_{t-1} \mid x_t)$ | Reverse SDE |
| Predicting noise $\varepsilon_\theta$ | Predicting score $s_\theta$ |
| DDPM sampling steps | Euler-Maruyama discretization |
| DDIM (deterministic) | Probability flow ODE |

**Key insight**: DDPM is the **discretized version** of the VP-SDE with specific choices of $f$ and $g$.

**Specifically:**

- DDPM uses $\beta_t$ schedule
- This corresponds to VP-SDE with:
  - $f(x,t) = -\frac{1}{2}\beta(t) x$
  - $g(t) = \sqrt{\beta(t)}$

**Nothing fundamentally new—just a cleaner, more general lens.**

---

## 7. Concrete Example: VP-SDE

Let's make everything explicit with the **Variance-Preserving SDE**.

### Forward Process

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw(t)
$$

**Why "variance-preserving"?**
The drift term $-\frac{1}{2}\beta(t) x$ exactly balances the diffusion to keep $\mathbb{E}[\|x_t\|^2]$ constant.

### Marginal Distribution

The transition distribution has closed form:

$$
p_t(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I\right)
$$

where:

$$

\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

**Sampling $x_t$:**
$$

x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,I)
$$

### Score Function

The conditional score is:

$$
\nabla_x \log p_t(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0}{1 - \bar{\alpha}_t} = -\frac{\varepsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

### Reverse SDE

$$
dx = \left[-\frac{1}{2}\beta(t) x - \beta(t) s_\theta(x,t)\right] dt + \sqrt{\beta(t)}\,d\bar{w}(t)
$$

### Training Loss

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\left[(1-\bar{\alpha}_t) \left\| s_\theta(x_t, t) + \frac{\varepsilon}{\sqrt{1-\bar{\alpha}_t}} \right\|^2\right]
$$

**This is exactly the DDPM loss** (up to weighting).

---

## Summary

**Core concepts:**
1. An **SDE** describes continuous-time random evolution: $dx = f(x,t)\,dt + g(t)\,dw(t)$
2. **Drift** $f(x,t)$: Deterministic flow (chosen by you)
3. **Diffusion** $g(t)$: Noise strength (chosen by you)
4. **Brownian motion** $w(t)$: Source of randomness
5. **Score** $\nabla_x \log p_t(x)$: The **only learned object**

**Training = Learning the score at all noise levels**

**Sampling = Solving the reverse SDE numerically**

**Next steps:**

- See [`02_sde_formulation.ipynb`](./02_sde_formulation.ipynb) for code implementation
- Study specific SDEs: VP-SDE, VE-SDE, sub-VP-SDE
- Learn about probability flow ODEs and fast sampling
- Apply to scPPDM (latent-space VP-SDE for drug response)

---

## References

- **Song et al. (2021)**: [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- **Ho et al. (2020)**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **Anderson (1982)**: Reverse-time diffusion equation models
- **Øksendal (2003)**: Stochastic Differential Equations: An Introduction with Applications
