# Diffusion Transformers (DiT): A Tutorial

This tutorial explains **Diffusion Transformers (DiT)** — the architectural shift from convolutional U-Nets to Transformers for generative modeling. We cover why this shift happened, how DiT works, and why it generalizes beyond images.

**Prerequisites**: Familiarity with rectified flow or diffusion models (see `docs/flow_matching/rectifying_flow.md`).

---

## 1. What is a Diffusion Transformer?

A **Diffusion Transformer (DiT)** is not a new diffusion theory — it's an **architectural choice**.

DiT is simply a Transformer used to parameterize the function learned in diffusion or flow-based models:

$$
v_\theta(x, t, c) \quad \text{(velocity prediction for rectified flow)}
$$

$$
\varepsilon_\theta(x, t, c) \quad \text{(noise prediction for DDPM)}
$$

$$
s_\theta(x, t) \quad \text{(score prediction for score matching)}
$$

The **objective** (what to learn) and the **architecture** (how to learn it) are orthogonal design choices.

---

## 2. Why U-Nets Dominated Early Diffusion

Historically, diffusion models used **U-Net** architectures because:

| Strength | Why It Helped |
|----------|---------------|
| Local structure | Images have strong spatial correlations |
| Multiscale features | Downsampling captures global context |
| Efficient | Convolutions are fast and well-optimized |
| Inductive bias | Spatial structure is built into the architecture |

**U-Net learns:**

- Local interactions in early layers
- Global interactions via progressive downsampling
- Skip connections preserve fine details

This worked extremely well for images, but came with limitations:

- **Fixed grid assumptions**: Inputs must be regular grids
- **Awkward conditioning**: Adding new conditions requires architectural changes
- **Limited flexibility**: Hard to apply to non-image data
- **Special handling**: Time and modality need custom integration

---

## 3. The Architectural Shift: Grids → Tokens

Transformers operate on **tokens**, not grids. The key conceptual move in DiT:

> Represent the input $x_t$ as a **sequence of tokens**.

**For images:**

1. Split image into patches (e.g., 16×16 pixels)
2. Flatten each patch into a vector
3. Embed into token space

**For other domains:**

- Genes, cells, regions, timepoints → tokens
- Patches are a metaphor, not a requirement

---

## 4. Input Representation

Let $x_t \in \mathbb{R}^d$ be the noisy (or interpolated) input at time $t$.

**Tokenization:**

$$
X_t = [x_t^{(1)}, x_t^{(2)}, \ldots, x_t^{(N)}]
$$

where:

- $x_t^{(i)} \in \mathbb{R}^{d_{\text{patch}}}$ is the $i$-th patch
- $N$ is the number of tokens

**Embedding:**

$$
h^{(i)} = W_{\text{embed}} \cdot x_t^{(i)} + e^{(i)}_{\text{pos}}
$$

The Transformer input is:

$$
H = [h^{(1)}, \ldots, h^{(N)}]
$$

---

## 5. Time Conditioning via Adaptive LayerNorm

Diffusion models are **time-conditioned**. DiT handles this elegantly through **modulation**, not concatenation.

**Standard Transformer block:**

$$
\text{Block}(H) = \text{MLP}(\text{Attention}(\text{LN}(H)))
$$

**DiT with Adaptive LayerNorm (AdaLN):**

$$
\text{AdaLN}(h, t) = \gamma(t) \cdot \text{LN}(h) + \beta(t)
$$

where $\gamma(t)$ and $\beta(t)$ are produced from a time embedding:

$$
\tau = \text{TimeEmbed}(t) \quad \rightarrow \quad (\gamma, \beta) = \text{MLP}(\tau)
$$

> **Deep dive**: For a detailed explanation of how time embeddings work and why the MLP doesn't "perturb ordering," see [time_embeddings_explained.md](time_embeddings_explained.md).

**Key insight**: Time controls the *behavior* of the network at every layer, not just its input.

This is the **FiLM (Feature-wise Linear Modulation)** pattern, which is much cleaner than concatenating $t$ to inputs.

---

## 6. Conditioning Beyond Time

The same AdaLN mechanism handles arbitrary conditions:

- Class labels
- Text embeddings
- Perturbation tokens
- Experimental conditions

**Two approaches:**

1. **Modulation**: Embed condition $c \mapsto e_c$, use for AdaLN parameters
2. **Cross-attention**: Append condition tokens, attend to them

Transformers make adding new conditions trivial — no architectural surgery required.

---

## 7. What the Transformer Computes

Inside the Transformer:

$$
H_{\text{out}} = \text{Transformer}(H_{\text{in}}, t, c)
$$

Then project back to output space:

$$
v_\theta(x_t, t, c) = W_{\text{out}} \cdot H_{\text{out}}
$$

**Conceptually:**

- **Self-attention**: Learns global dependencies between all tokens
- **MLPs**: Refine local nonlinearities
- **Time modulation**: Tells the network where it is along the trajectory

This works regardless of training objective (score matching, noise prediction, or rectified flow).

---

## 8. DiT + Rectified Flow

Combining DiT with rectified flow is particularly elegant.

**Recall rectified flow target:**

$$
\text{target} = x_1 - x_0
$$

**DiT training loss:**

$$
\mathcal{L} = \mathbb{E}_{x_0, x_1, t} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

where:

- $v_\theta$ is a Transformer
- $x_t$ is tokenized
- $t$ modulates every layer via AdaLN

**Why this combination works well:**

| Component | Contribution |
|-----------|--------------|
| Transformers | Model long-range structure via attention |
| Rectified flow | Simple, stable regression target |
| AdaLN | Clean time/condition integration |
| ODE sampling | Fast, deterministic generation |

---

## 9. Why DiT Scales Better Than U-Net

Three structural reasons:

### Global Context is Native

Self-attention is global by default. No need for deep pyramids to propagate information across the image.

### Shape Flexibility

With packing/masking tricks (Patch-n-Pack):

- Variable image sizes in same batch
- Variable video lengths
- Heterogeneous biological objects

This is impossible to do cleanly with CNNs.

### Conditioning is First-Class

Adding a new condition:

- Add tokens, or
- Add modulation parameters

No architectural changes needed.

---

## 10. Beyond Images: DiT as a General Engine

Once you think of DiT as:

> "A Transformer learning a time-dependent vector field"

It becomes a **general-purpose continuous generative engine**.

**Applications:**

- Images (Stable Diffusion 3, DALL-E 3)
- Videos (Sora, Goku)
- Audio (AudioLDM)
- Molecules (protein structure)
- Trajectories (robotics)
- Latent biological states (gene expression)

**Key insight:**

- Rectified flow removes density assumptions
- Transformers remove grid assumptions
- Together, they're highly portable

---

## 11. Summary

> **A Diffusion Transformer is a Transformer trained to predict time-conditioned vector fields, replacing convolutional inductive bias with global token interaction.**

**Key components:**

| Component | Purpose |
|-----------|---------|
| Patch embedding | Convert input to tokens |
| Positional encoding | Preserve spatial/sequential structure |
| AdaLN | Time and condition modulation |
| Self-attention | Global dependencies |
| Output projection | Map back to target space |

**The modern generative stack:**

```
Rectified Flow (objective) + DiT (architecture) + AdaLN (conditioning)
```

---

## References

- Peebles & Xie (2023) - "Scalable Diffusion Models with Transformers" (DiT paper)
- Perez et al. (2018) - "FiLM: Visual Reasoning with a General Conditioning Layer"
- Dosovitskiy et al. (2020) - "An Image is Worth 16x16 Words" (ViT)

---

# Advanced Topics: Alternative Backbones for Biology

The following sections explore alternatives to Transformers for biological applications, where tokenization may not be natural.

---

## 12. What Diffusion Actually Requires from a Backbone

Strip away the branding. A diffusion or rectified-flow model needs a function:

$$
f_\theta(x_t, t, c) \rightarrow \text{vector field}
$$

**Requirements:**

- Accept a state representation
- Condition on time
- Optionally condition on context
- Output a vector of the same dimensionality as the state

**The real requirement:**

> A model capable of learning global dependencies and time-conditioned transformations.

Transformers satisfy this — but they are not unique.

---

## 13. State-Space Models as Diffusion Backbones

**Can SSMs (Mamba, S4) or long convolutions (Hyena) be diffusion backbones?**

Yes. In fact, this is a **natural pairing**.

**Why?**

- Rectified flow defines **continuous-time dynamics**
- State-space models are *literally designed* to model dynamics

Architectures like:

- Long convolution models
- SSMs (S4, Mamba)
- Hyena-style implicit sequence operators

are philosophically aligned with flow-based generative modeling.

**Why Transformers won historically:**

- Easy to scale
- Clean conditioning via cross-attention
- Unified modalities early
- Infrastructure exists

But this is historical inertia, not a fundamental requirement.

---

## 14. The Tokenization Problem for Gene Expression

Gene expression vectors:

$$
x \in \mathbb{R}^{G}
$$

where $G$ is the number of genes.

**Properties:**

- Unordered (no natural sequence)
- Dense (most genes have non-zero expression)
- Compositional (relative, not absolute)
- Population-relative

**The problem with "genes as tokens":**

Approaches like Geneformer rank genes by expression and treat them as a sequence. This *works*, but feels **ontologically wrong**:

> Ranking genes is not a natural ordering of biological state — it's an engineering trick.

---

## 15. Better Representations for Gene Expression

### Option A: State Vector (No Tokens)

Treat expression as a single state vector:

- $x_t \in \mathbb{R}^G$
- Backbone: MLP, SSM, or continuous-time operator
- Time-conditioning via FiLM

This aligns beautifully with rectified flow — you're learning a velocity in gene-expression space.

### Option B: Latent-Space Diffusion

Instead of tokenizing raw expression:

1. Encode expression into latent state $z \in \mathbb{R}^d$
2. Run diffusion/rectified flow in latent space
3. Decode only if necessary

The backbone sees:

- Smooth, lower-dimensional states
- No artificial ordering
- No sparsity pathologies

This is where JEPA, VAEs, and diffusion naturally converge.

### Option C: Set-Based Representations

If you insist on tokens, do it honestly:

- Represent expression as a **set** (unordered)
- Genes have embeddings
- Expression value modulates them
- Use permutation-invariant operators
- Attention without positional encoding

### Option D: Dynamics-First (SSM-Friendly)

If your data is time-series, perturb-seq, or trajectories:

- The sequence is **time**, not genes
- Each timestep holds a full expression state or latent
- Backbone models temporal evolution

This is where SSMs and Hyena-style operators shine.

---

## 16. A Natural Architecture for Perturb-Seq

Combining the insights above:

```
Expression → Encoder → Latent State
                          ↓
              SSM/Hyena modeling latent dynamics
                          ↓
              Rectified flow in latent space
                          ↓
              Decoder → Expression (if needed)
```

**Properties:**

- No fake tokens
- No gene ranking
- Natural temporal modeling
- Proper count handling via VAE decoder

---

## 17. The Organizing Principle

> **Tokenization is a convenience for architectures, not a requirement of the data.**

Once you internalize this:

- DiT becomes "Transformer-as-backbone"
- Rectified flow becomes "state evolution"
- Hyena/SSMs become first-class alternatives
- Gene expression stops being forced into unnatural formats

---

## Future Directions

See `docs/incubation/` for explorations of:

- Latent rectified-flow + SSM architectures for perturb-seq
- Transformer vs SSM inductive biases for biological dynamics
- When tokenization *is* biologically meaningful (pathways, modules)
