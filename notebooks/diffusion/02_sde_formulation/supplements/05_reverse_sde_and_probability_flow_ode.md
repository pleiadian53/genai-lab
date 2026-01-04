# Understanding the Reverse SDE and Probability Flow ODE

## The Reverse-Time SDE

### The Formula

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,dw
$$

### Term-by-Term Interpretation

The infinitesimal change in $x$ (the data) is the sum of:

1. **$f(x,t)\,dt$** — The original drift from the forward process
   - This term "continues" the deterministic flow from the forward SDE
   - In VP-SDE: $f(x,t) = -\frac{1}{2}\beta(t)x$ (shrinks toward origin)
   - In VE-SDE: $f(x,t) = 0$ (no drift)

2. **$-g(t)^2 \nabla_x \log p_t(x)\,dt$** — The **score correction** (the key term!)
   - Points toward regions of higher probability density
   - $\nabla_x \log p_t(x)$ is the score function: "which direction makes $x$ more likely?"
   - Scaled by $g(t)^2$: stronger correction when noise is high
   - **This term reverses the diffusion**: it "undoes" the noise

3. **$g(t)\,dw$** — Stochastic noise (Brownian motion)
   - Same diffusion coefficient as the forward process
   - Maintains diversity in generated samples
   - $dw$ is a reverse-time Brownian increment

### Intuition

Think of it as:
$$
dx = \underbrace{f(x,t)\,dt}_{\text{continue forward drift}} + \underbrace{(-g(t)^2 \nabla_x \log p_t(x))\,dt}_{\text{denoise: move toward data}} + \underbrace{g(t)\,dw}_{\text{add randomness}}
$$

The score term is what makes generation possible. Without it, we'd just be running the forward process in reverse time, which wouldn't recover data from noise.

---

## Where Does the Probability Flow ODE Come From?

### The Formula

$$
dx = \left[f(x,t) - \tfrac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt
$$

Note: **No $dw$ term** — this is deterministic!

### The Origin: Anderson's Result (1982)

The probability flow ODE comes from a deep result in stochastic calculus:

**Key theorem (Anderson 1982)**: For any SDE, there exists a corresponding ODE that produces the **same marginal distributions** $p_t(x)$ at each time $t$.

This is not obvious! The SDE has randomness; the ODE has none. Yet they produce the same distribution over states.

### The Derivation (High-Level)

1. **Start with the reverse SDE**:
   $$
   dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,dw
   $$

2. **The Fokker-Planck equation** describes how probability density evolves:
   $$
   \frac{\partial p_t}{\partial t} = -\nabla \cdot (f \cdot p_t) + \frac{1}{2}g(t)^2 \nabla^2 p_t
   $$

3. **Key insight**: The diffusion term $\frac{1}{2}g(t)^2 \nabla^2 p_t$ can be rewritten as a drift term involving the score:
   $$
   \frac{1}{2}g(t)^2 \nabla^2 p_t = \nabla \cdot \left(\frac{1}{2}g(t)^2 \nabla p_t\right) = \nabla \cdot \left(\frac{1}{2}g(t)^2 p_t \nabla \log p_t\right)
   $$

4. **Absorb diffusion into drift**: Combine the original drift with this "effective drift" from diffusion:
   $$
   \text{Effective drift} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)
   $$

5. **Result**: An ODE with this effective drift produces the **same** Fokker-Planck equation, hence the same $p_t(x)$.

### Why Factor of 1/2?

Compare the two:

| Equation | Score coefficient | Has noise? |
|----------|-------------------|------------|
| Reverse SDE | $g(t)^2$ | Yes ($+g(t)dw$) |
| Probability Flow ODE | $\frac{1}{2}g(t)^2$ | No |

The factor of $\frac{1}{2}$ compensates for the missing stochastic term. Roughly:
- In the SDE, noise contributes to spreading probability mass
- In the ODE, the score term must do **all** the work of shaping the distribution
- But without noise, you need less "pull" from the score (hence $\frac{1}{2}$)

---

## Connection Between SDE and ODE

### Same Marginals, Different Paths

Both equations produce samples from the same distribution $p_0(x)$ (data distribution), but:

| Property | Reverse SDE | Probability Flow ODE |
|----------|-------------|---------------------|
| **Paths** | Random (different each run) | Deterministic (same every time) |
| **Diversity** | High (noise adds variety) | Lower (same input → same output) |
| **Speed** | Slower (needs small $dt$) | Faster (can use larger steps) |
| **Interpolation** | Not meaningful | Latent space is well-defined |

### Why Have Both?

**SDE sampling**:
- More diverse outputs
- Better for creative generation
- Theoretically "correct" reverse process

**ODE sampling**:
- Faster (DDIM is based on this)
- Deterministic: good for reproducibility
- Enables latent space manipulation
- Can use advanced ODE solvers (Runge-Kutta, etc.)

---

## Practical Implications

### For DDPM/DDIM

- **DDPM**: Discretized reverse SDE (with noise)
- **DDIM**: Discretized probability flow ODE (deterministic)
- Both use the same trained model $s_\theta(x,t)$

### For Fast Sampling

ODE formulation enables:
- Larger time steps (less sensitive to discretization)
- Adaptive step-size solvers
- Fewer neural network evaluations (faster generation)

### The Score Is Everything

In both formulations, **the score function $\nabla_x \log p_t(x)$ is the only learned quantity**. Once you have it, you can choose either:
- Stochastic sampling (SDE)
- Deterministic sampling (ODE)

---

## Summary

| Question | Answer |
|----------|--------|
| What does the reverse SDE compute? | $dx =$ (continue drift) + (denoise via score) + (add noise) |
| Where does the ODE come from? | Fokker-Planck: absorb diffusion into an effective drift |
| Why factor of 1/2? | Compensates for missing stochastic term |
| Which to use? | SDE for diversity, ODE for speed/determinism |

---

## References

- **Anderson (1982)**: "Reverse-time diffusion equation models" — Original derivation of reverse-time dynamics
- **Song et al. (2021)**: [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — Modern treatment with probability flow ODE
- **Ho et al. (2020)**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM as discretized SDE

