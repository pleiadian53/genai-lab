# SDE Formulation: Common Questions Answered

This document addresses frequently asked questions about the SDE (Stochastic Differential Equation) formulation of diffusion models. It complements the main tutorial by diving deeper into practical and conceptual questions.

**Prerequisites**: Basic understanding of SDEs. See [`sde_formulation.md`](./sde_formulation.md) for foundations.

---

## Quick Recap: What is an SDE?

A **stochastic differential equation (SDE)** adds *continuous random noise* to deterministic motion:

$$
dx(t) = f(x(t), t)\,dt + g(t)\,dw(t)
$$

This describes a **random process evolving in time**, not a single deterministic trajectory.

**Components**:

- $f(x,t)$: Drift (deterministic flow)
- $g(t)$: Diffusion coefficient (noise strength)
- $dw(t)$: Brownian motion increment

---

## 1. How is an SDE System Solved?

### Short Answer

**Numerically. Always.**

There are essentially no closed-form solutions for the SDEs used in diffusion models. Unlike simple ODEs where you might write $x(t) = x_0 e^{-\lambda t}$, SDEs require numerical simulation.

### What "Solving an SDE" Actually Means

When we say "solve an SDE," we mean **simulating sample paths** of the random process $x(t)$.

**Conceptually**: Starting from an initial state $x_0$, we step forward (or backward) in tiny time increments, adding both:
1. **Deterministic drift**: Where the system "wants" to go
2. **Random diffusion**: Noise that perturbs the path

Each simulation produces one random trajectory. Run it 1000 times, get 1000 different paths—all following the same SDE.

---

### The Euler–Maruyama Method (Basic Solver)

For the SDE:

$$
dx(t) = f(x(t),t)\,dt + g(t)\,dw(t)
$$

**Euler–Maruyama** discretizes time into steps of size $\Delta t$:

$$
x_{k+1} = x_k + f(x_k,t_k)\,\Delta t + g(t_k)\sqrt{\Delta t}\,\varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0,I)
$$

**Interpretation**:

- **Deterministic motion**: $f(x_k,t_k)\Delta t$ — where drift pushes you
- **Stochastic motion**: $g(t_k)\sqrt{\Delta t}\,\varepsilon_k$ — random kick from noise

**Key insight**: Noise scales as $\sqrt{\Delta t}$, not $\Delta t$. This is fundamental to Brownian motion.

This is the **SDE analogue of Euler's method** for ODEs, but with added randomness at each step.

---

### In Diffusion Models Specifically

#### Forward Process (Data → Noise)

The forward corruption process can be handled in two ways:

1. **Analytically** (preferred): Use closed-form marginal distribution
   $$
   x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon
   $$

   This is exact and fast—no need to simulate step-by-step.

2. **Numerically** (rare): Simulate via Euler–Maruyama
   - Only needed for exotic SDEs without closed forms
   - Slower and less accurate

**In practice**: We almost always use the closed-form marginal during training.

#### Reverse Process (Noise → Data)

The reverse process for **generation** is always solved numerically:

**Common methods**:

- **Euler–Maruyama**: Simple, first-order
- **Predictor–corrector**: Alternate between drift step and Langevin correction
- **Higher-order solvers**: Heun, Runge-Kutta (better accuracy, fewer steps)
- **ODE solvers**: For deterministic sampling (see below)

**Why numerical?** The reverse SDE depends on the learned score function $s_\theta(x,t)$, which is a neural network—no closed form exists.

---

### Probability Flow ODE (Important Special Case)

Here's a remarkable fact: The reverse SDE:

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,dw
$$

has an **ODE cousin** with the **same marginal distributions**:

$$
dx = \left[f(x,t) - \tfrac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt
$$

**Key differences**:

| Property | SDE | ODE |
|----------|-----|-----|
| Randomness | Stochastic ($+g(t)dw$) | Deterministic (no noise) |
| Paths | Different each run | Same path every time |
| Speed | Slower (needs small steps) | Faster (larger steps OK) |
| Diversity | Higher sample diversity | Lower diversity |

**Practical implications**:

- **ODE sampling** underlies DDIM and fast samplers
- **SDE sampling** gives more diverse outputs
- Both generate from the same distribution $p_0(x)$

So diffusion models can be sampled **stochastically (SDE)** or **deterministically (ODE)**—your choice!

---

## 2. What Models Are Learned in the SDE Formulation?

This is the most important conceptual question. Let's be crystal clear about what's fixed versus what's learned.

### What is NOT Learned

You do **not** learn:

- The SDE itself
- The drift function $f(x,t)$
- The diffusion coefficient $g(t)$
- The Wiener process $w(t)$

These are all **design choices** you make upfront. They define the corruption process but contain no learnable parameters.

**Why this matters**: Many people mistakenly think the neural network learns "how to add noise." It doesn't. The noise schedule is fixed. The network learns something else entirely.

---

### What IS Learned (The Only Thing)

$$
\boxed{
s_\theta(x,t) \approx \nabla_x \log p_t(x)
}
$$

This is called the **score function**.

**Interpretation**:

- **Geometrically**: Direction of steepest increase in log probability
- **Intuitively**: Vector field pointing toward "more data-like" regions
- **Practically**: Tells you which way to move to denoise the data

**Dimensionality**: If your data is $x \in \mathbb{R}^d$, the score is also a vector in $\mathbb{R}^d$. For images, that's millions of dimensions—one gradient component per pixel.

---

### What Do We Use the Learned Score For?

This is crucial to understand. The score function $s_\theta(x,t)$ is used for **sampling** (generation).

**During sampling**, we solve the reverse-time SDE:

$$
dx = \left[f(x,t) - g(t)^2 s_\theta(x,t)\right]dt + g(t)\,dw
$$

At each step:
1. **Evaluate** the score: $s_\theta(x_t, t)$ tells us which direction increases probability
2. **Drift toward data**: The term $-g(t)^2 s_\theta(x,t)$ pulls us toward high-probability regions
3. **Add noise**: The term $g(t)dw$ maintains diversity

**The score is the bridge** between noise and data. Without it, we couldn't reverse the diffusion process.

**Analogy**: Imagine you're lost in fog (noise). The score function is like a compass that always points toward civilization (data). By following it and taking small steps, you gradually emerge from the fog.

---

### Neural Network Architecture

**Input**:

- Noisy data: $x_t \in \mathbb{R}^d$
- Time/noise level: $t \in [0,T]$ (usually embedded as sinusoidal features)

**Output**:

A vector in $\mathbb{R}^d$ representing one of these **equivalent parameterizations**:

1. **Score**: $s_\theta(x_t,t) \approx \nabla_x \log p_t(x_t)$
2. **Noise**: $\varepsilon_\theta(x_t,t) \approx \varepsilon$ (the noise that was added)
3. **Clean data**: $\hat{x}_0 \approx x_0$ (denoised prediction)

These are mathematically equivalent—you can convert between them using the forward process equations.

---

### Why Predicting Noise Works

For Gaussian corruption with forward process:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon
$$

The conditional score has a closed form:

$$
\nabla_x \log p_t(x_t \mid x_0) = -\frac{\varepsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

**Key insight**: The score is just the noise, scaled by $-1/\sigma_t$.

So:

- **Predicting noise** $\varepsilon_\theta$
- **Predicting score** $s_\theta$
- **Predicting clean data** $\hat{x}_0$

are all the same signal, just scaled/shifted differently. DDPM predicts noise, score-based models predict the score, but they're equivalent.

---

### Training Workflow (SDE View)

Here's the complete training loop:

1. **Sample clean data**: $x_0 \sim p_{\text{data}}$
2. **Sample time**: $t \sim \text{Uniform}(0,T)$
3. **Sample noise**: $\varepsilon \sim \mathcal{N}(0,I)$
4. **Generate noisy data**: $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon$ (closed-form marginal)
5. **Predict score/noise**: $s_\theta(x_t, t)$ or $\varepsilon_\theta(x_t, t)$
6. **Compute loss**: $\mathcal{L} = \|s_\theta(x_t,t) - (-\varepsilon/\sigma_t)\|^2$ (or equivalent)
7. **Backpropagate**: Update $\theta$

**Crucial observation**: No SDE solving during training! We use the closed-form marginal to generate noisy samples directly. SDE solving only happens during sampling/generation.

---

## 3. Is Brownian Motion the Only Way to Model Randomness?

### Short Answer

**No. But it's the only one used in diffusion models (so far).**

Let's separate mathematical theory from practical machine learning.

---

### Why Brownian Motion is Used in Diffusion Models

Brownian motion (Wiener process) has unique mathematical properties that make diffusion models tractable:

**Mathematical properties**:

- **Continuous paths**: No sudden jumps, smooth evolution
- **Gaussian increments**: $w(t+\Delta t) - w(t) \sim \mathcal{N}(0, \Delta t)$
- **Markov property**: Future depends only on present, not past
- **Independent increments**: Non-overlapping intervals are independent

**Why these matter for diffusion models**:

- **Exact reverse SDE**: Anderson (1982) proved that Brownian SDEs have tractable reverse-time equations
- **Clean score formulation**: The score $\nabla_x \log p_t(x)$ has a well-defined meaning
- **Stable training**: Gaussian noise is well-behaved, no heavy tails or pathological cases
- **Closed-form marginals**: For many SDEs (like VP-SDE), we can compute $p_t(x|x_0)$ analytically

**Bottom line**: Brownian motion gives us mathematical control. We can derive, train, and sample reliably.

---

### Other Stochastic Processes in SDEs (Finance, Physics)

You're absolutely right that algorithmic trading and quantitative finance use many other stochastic processes. Here are the main alternatives:

#### 1. Jump Processes (Lévy Processes)

**SDE form**:

$$
dx = f(x,t)\,dt + \sigma\,dW_t + dJ_t
$$

where $J_t$ is a jump process (e.g., compound Poisson).

**Characteristics**:

- **Sudden jumps**: Discontinuous paths
- **Heavy tails**: Captures extreme events
- **Market crashes**: Models rare but large moves

**Examples**:

- **Poisson jumps**: Fixed-size jumps at random times
- **Variance Gamma**: Infinite activity, finite variation
- **CGMY models**: Captures both small and large jumps

**Why not in diffusion models?** Reverse-time equations for jump processes are much more complex. Score matching becomes ill-defined at jump points.

---

#### 2. Stochastic Volatility Models

**Example (Heston model)**:

$$
\begin{aligned}
dS_t &= \mu S_t\,dt + \sqrt{v_t} S_t\,dW_t \\
dv_t &= \kappa(\theta - v_t)\,dt + \xi \sqrt{v_t}\,dB_t
\end{aligned}
$$

**Characteristics**:

- **Randomness in randomness**: Volatility itself is stochastic
- **Two coupled SDEs**: State and volatility evolve together
- **Volatility clustering**: Periods of high/low volatility persist

**Why not in diffusion models?** Would require learning a time-varying diffusion coefficient $g(x,t)$, significantly complicating the model.

---

#### 3. Fractional Brownian Motion (fBm)

**Characteristics**:

- **Long-range dependence**: Past affects future over long horizons
- **Non-Markovian**: Violates the Markov property
- **Hurst exponent**: $H \in (0,1)$ controls roughness
  - $H = 0.5$: Standard Brownian motion
  - $H < 0.5$: Rough, mean-reverting
  - $H > 0.5$: Smooth, trending

**Applications**: Rough volatility models in finance, network traffic

**Why not in diffusion models?** Non-Markovian processes don't have simple reverse-time SDEs. The score function would need to depend on the entire history, not just current state.

---

#### 4. Colored Noise

**Characteristics**:

- **Correlated increments**: $\text{Cov}(dw_t, dw_s) \neq 0$ for $t \neq s$
- **Violates white-noise assumption**: Brownian motion has "white" spectrum
- **Frequency-dependent**: Different noise at different timescales

**Applications**: Physical systems with memory, environmental noise

**Why not in diffusion models?** Breaks the mathematical framework. Anderson's reverse-time theorem assumes white noise.

---

### Why Diffusion Models Don't Use These (Yet)

The fundamental issue is **tractability of reverse-time dynamics**.

**Problems with non-Brownian noise**:

1. **Reverse-time equations become messy or unknown**: No clean formula like Anderson's theorem
2. **Score matching may be ill-defined**: What is $\nabla_x \log p_t(x)$ at a jump?
3. **Sampling becomes unstable**: Numerical solvers for exotic SDEs are less reliable
4. **No closed-form marginals**: Can't efficiently generate training samples

**The trade-off**: Diffusion models sacrifice realism of noise for **mathematical control**. Brownian motion is "boring" but tractable.

**Future research**: Some recent work explores:
- **Lévy diffusion models**: Incorporating small jumps
- **Adaptive noise schedules**: Learning $g(t)$ instead of fixing it
- **Non-Markovian extensions**: Using neural ODEs with memory

But these are still experimental and not widely adopted.

---

## Summary: The Big Picture

Let's synthesize everything into a coherent view:

**Core principles**:

1. **An SDE defines how probability mass flows over time**: From data to noise (forward) and back (reverse)
2. **The forward SDE is fixed and simple**: You choose $f(x,t)$ and $g(t)$ upfront
3. **The only learned object is the score**: $s_\theta(x,t) \approx \nabla_x \log p_t(x)$, a time-dependent vector field
4. **Sampling is numerical integration**: Solve the reverse SDE using Euler-Maruyama or ODE solvers
5. **Brownian motion enables tractability**: Reverse-time theory, score matching, and stable training

**Why this framework is powerful**:

- **Continuous-time**: More general than discrete DDPM
- **Unified**: Score-based models, DDPM, and DDIM are all special cases
- **Flexible**: Can design custom SDEs for specific applications
- **Interpretable**: Clear separation between design choices and learning

**Next steps for deeper understanding**:

1. Take a concrete SDE (e.g., VP-SDE)
2. Write down $f(x,t)$ and $g(t)$ explicitly
3. Derive the closed-form marginal $p_t(x|x_0)$
4. Discretize the reverse SDE into update rules
5. Implement it in code (see [`02_sde_formulation.ipynb`](./02_sde_formulation.ipynb))

That's where everything clicks and stops being abstract. The math becomes concrete, and you can see exactly how DDPM emerges from the SDE formulation.

---

## Further Reading

- **Song et al. (2021)**: [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — The definitive paper
- **Anderson (1982)**: Reverse-time diffusion equation models — Original reverse-time theorem
- **Øksendal (2003)**: Stochastic Differential Equations — Comprehensive textbook
- **Karatzas & Shreve (1991)**: Brownian Motion and Stochastic Calculus — Advanced reference
