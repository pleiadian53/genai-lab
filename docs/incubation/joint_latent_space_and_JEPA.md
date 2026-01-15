# Joint Latent Spaces and JEPA: A Unified Approach for Computational Biology

## Introduction

This document explores a powerful architectural idea emerging from the vision community that has profound implications for computational biology: **joint latent spaces** and **Joint Embedding Predictive Architecture (JEPA)**.

The core insight is deceptively simple:

> If two data types differ only by dimensionality or observation density, they probably want the same latent space.

We'll trace this idea from its origins in joint image-video generation (specifically, ByteDance's Goku model) through to practical architectures for biological applications like Perturb-seq, single-cell analysis, and multi-omics integration.

---

## Part 1: The Goku Model — Where This Idea Crystallized

### The Provocation

ByteDance and HKU researchers asked a simple question: why do we treat images and videos as fundamentally different, when mathematically they're almost the same object?

An image is just a video with **time dimension = 1**:

$$
\text{Video} \in \mathbb{R}^{T \times H \times W \times C}
$$

An image is simply $T = 1$.

The historical split happened for engineering reasons—images were easier, videos expensive, temporal modeling scary. So we trained separate systems, even though the underlying inductive bias mismatch was artificial.

**The biological parallel**: This is the same historical accident as treating genes vs. transcripts vs. isoforms as separate modeling problems, when they're just different *slices* of the same biological process.

### The Solution: One Latent Space to Rule Them All

Goku's first conceptual move isn't about diffusion—it's the **joint VAE**.

Instead of separate encoders:
- Image VAE → image latent space
- Video VAE → video latent space

They use **one encoder-decoder pair** that maps both images and videos into the **same latent manifold**.

**Why this matters:**

- Image-only data teaches **spatial priors** (texture, composition, objects)
- Video data teaches **dynamics** (motion, continuity, causality)
- Both shape the *same latent geometry*

This is a powerful prior-sharing mechanism.

### Combio Analogy

Think about this mapping:
- **Bulk RNA-seq** ≈ "static image"
- **Time-series, Perturb-seq, lineage tracing** ≈ "video"
- **Single-cell snapshots across conditions** ≈ variable-length clips

A joint latent space means static and dynamic biology train each other instead of living in silos.

---

## Part 2: Key Architectural Components

### 2.1 Rectified Flow (Instead of Classic Diffusion)

Goku uses **rectified flow** rather than classic DDPM noise schedules.

**Conceptual difference:**

- **Diffusion**: Learn to undo noise step by step
- **Rectified flow**: Learn a *direct velocity field* that moves noise → data

**Why this matters for joint training:**

- Flow matching is simpler to condition
- It handles variable sequence lengths more gracefully
- Less bookkeeping when mixing modalities

This is particularly appealing for **non-image domains** like biology, where the "noise semantics" are already fuzzy. What does "adding Gaussian noise to gene expression" really mean biologically? Flow-based approaches sidestep this question.

### 2.2 Patch n' Pack: Batching Without Caring About Shapes

This is perhaps the most under-appreciated idea.

**Traditional approach:**

- Pad videos to the same length
- Resize everything to the same resolution
- Waste computation on padding tokens

**Patch n' Pack approach:**

- Tokenize everything into patches
- Concatenate all tokens into one long sequence
- Add block attention masks so tokens only attend within their own sample

This lets a minibatch contain:
- Images of different sizes
- Videos of different lengths
- All mixed together

The philosophical insight:

> Shape uniformity is not a law of nature—it's a convenience hack.

**Combio parallel**: You routinely deal with:
- Transcripts of different lengths
- Genes with variable exon counts
- Cells with variable detected features
- Perturbations with different temporal spans

Patch n' Pack is the Transformer version of **respecting biological heterogeneity** instead of padding it away.

### 2.3 Full Attention (Not Factorized)

Most video models split attention into:
- Spatial attention
- Temporal attention

Goku doesn't. It uses **plain full attention** with masking.

**Why this matters:**

- No architectural bias saying "time is special"
- The model discovers when temporal relationships matter
- Images and videos become first-class citizens in the same token universe

This mirrors a recurring theme in computational biology: stop hard-coding "gene first, transcript later." Let representations *emerge* under constraints.

### 2.4 Positional Encoding Reset

RoPE positional encodings are reset **per sample block**.

This means:
- Token position 0 always means "start of *this* sample"
- Not "position 0 in a giant Frankenstein sequence"

**Biological parallel**: Each gene, each cell, each experiment has its own coordinate frame. Forcing a global coordinate system often destroys meaning.

---

## Part 3: The Connection to JEPA

### What is JEPA?

**Joint Embedding Predictive Architecture (JEPA)**, pioneered by Yann LeCun's team at Meta AI, makes a sharp philosophical move:

> Do not reconstruct pixels. Do not predict tokens. Predict **representations of future states**.

In JEPA:
1. You embed a *context* into latent space
2. You embed a *target* into latent space
3. You train a predictor so the context-latent can *predict* the target-latent

Crucially:
- The loss lives **entirely in latent space**
- The model never needs to care about raw observation noise

### The JEPA Family (2023-2025)

| Model | Date | Domain | Key Innovation |
|-------|------|--------|----------------|
| **I-JEPA** | Jun 2023 | Images | First JEPA; mask prediction in embedding space |
| **MC-JEPA** | Jul 2023 | Video | Motion + content disentanglement |
| **V-JEPA** | Feb 2024 | Video | Temporal prediction without language supervision |
| **V-JEPA 2** | Jun 2025 | Video + Robotics | World model for understanding, prediction, planning |
| **VL-JEPA** | Dec 2025 | Vision-Language | Predicts text embeddings, not tokens |

### V-JEPA 2: The Major 2025 Breakthrough

V-JEPA 2 is the first world model trained on video that enables:
- State-of-the-art video understanding
- Zero-shot physical prediction
- Robotic planning from observation

**Key findings:**

- Video encoder pre-trained *without* language supervision can be aligned with LLMs
- Self-supervised video pretraining enables physical world understanding
- Two-phase training: (1) self-supervised from video, (2) small amount of robot interaction data

### VL-JEPA: Vision-Language Efficiency

Instead of autoregressive token generation, VL-JEPA predicts **continuous text embeddings**.

**Advantages:**

- 50% fewer trainable parameters than standard VLMs
- Supports selective decoding (2.85× fewer decode operations)
- Natively supports open-vocabulary classification, retrieval, and VQA

### Why JEPA and Goku Share the Same Soul

Look at what Goku is doing conceptually:
- Image encoder → latent
- Video encoder → latent
- Diffusion/flow learns transitions *in latent space*
- Images are treated as degenerate videos ($T=1$)

**Same skeleton. Different skin.**

Both approaches reject a very old habit in ML: treating observations as the thing we should model directly. Instead, they assume:
- There exists an underlying state manifold
- Observations are partial, noisy, structured projections of that state
- Learning becomes easier if we operate *between states*, not pixels

### The Unifying Principle

> A joint latent space is where modality disappears, and JEPA is how time re-enters.

Different training mechanics. Same worldview.

---

## Part 4: JEPA for Computational Biology

### Why JEPA is a Natural Fit for Biology

By 2025-2026, JEPA-style thinking has won several arguments:
- Reconstruction losses are brittle
- Likelihoods are often misaligned with semantics
- Prediction in latent space scales better
- Masking + prediction beats full autoregression for long contexts

In computational biology, you almost never want:
- Perfect reconstruction of expression values
- Exact token-level likelihoods

You want:
- Correct *relationships*
- Plausible *state transitions*
- Meaningful *counterfactuals*

That is *exactly* the JEPA niche.

### A Unified Model for Static and Dynamic Biology

Here's a clean mental model that unifies everything:

**Joint encoder** maps:
- Static data (genome, baseline expression)
- Dynamic data (time series, perturbations)
- Multimodal data (RNA, ATAC, protein)

...into a shared latent state.

**JEPA-style predictor** learns:

$$
z_{t+\Delta} \approx f_\theta(z_t, \text{condition})
$$

- No reconstruction is strictly required
- Decoders are optional and task-specific
- Generative ability emerges from *state evolution*, not pixel synthesis

This is essentially a **biological world model**.

---

## Part 5: JEPA Architecture for Perturb-seq

Let's make this concrete with a sketch for Perturb-seq—one of the most powerful experimental technologies for understanding gene regulation.

### The Data

In Perturb-seq, per cell $i$ you have:
- $x_i$: gene expression vector (counts / log1p / normalized)
- $p_i$: perturbation label(s) (one-hot or multi-hot; e.g., sgRNA IDs)
- $c_i$: covariates (cell type, batch, donor, cell cycle, library size)
- Sometimes $t_i$: timepoint (if time-course)
- Optional: multiome (ATAC, protein) as extra modalities

### The JEPA Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                        JEPA for Perturb-seq                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                    ┌──────────────┐          │
│  │   Context    │                    │    Target    │          │
│  │   Encoder    │                    │   Encoder    │          │
│  │   E_ctx      │                    │  E_tgt (EMA) │          │
│  └──────┬───────┘                    └──────┬───────┘          │
│         │                                   │                   │
│         ▼                                   ▼                   │
│        h_i        ┌──────────────┐         z_i                 │
│    (context      │  Perturbation │    (target latent)          │
│     latent)      │    Encoder    │                             │
│         │        │    e(p_i)     │                             │
│         │        └──────┬────────┘                             │
│         │               │                                       │
│         ▼               ▼                                       │
│    ┌────────────────────────────┐                              │
│    │        Predictor F         │                              │
│    │   ẑ_i = F(h_i, e(p_i))    │───────── predict ──────→ z_i │
│    └────────────────────────────┘                              │
│                                                                 │
│    Loss: L = ||ẑ_i - sg(z_i)||² + variance/covariance reg     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### Target Encoder ($E_{\text{tgt}}$)

Updated via EMA (momentum). Takes the observed expression under perturbation and produces the "ground truth" state embedding.

- Input: $x_i$ (observed expression under perturbation $p_i$)
- Output: $z_i = E_{\text{tgt}}(x_i)$

#### Context Encoder ($E_{\text{ctx}}$)

Takes baseline context + covariates. The challenge: what is "baseline context" for a cell that's already perturbed?

**Options:**

- **Cell-type context**: Learned embedding of cell type/cluster label
- **Matched controls**: Sample a control cell from same covariates as baseline
- **Population baseline**: Covariate-conditioned baseline embedding (learned)

#### Perturbation Encoder

Represent perturbations as tokens:
- Single sgRNA: embedding lookup
- Multiple sgRNAs: set encoder (sum + MLP, or attention over perturbation tokens)

Output: $e(p_i)$

#### Predictor

Takes context embedding + perturbation embedding:

$$
\hat{z}_i = F(h_i, e(p_i))
$$

**Implementation options:**

- MLP with FiLM-style conditioning
- Cross-attention: "perturbation tokens attend to cell tokens"
- Small Transformer over tokens: $[h_i, e(p_i), e(c_i), e(t_i)]$ → 2-4 blocks → $\hat{z}$

### The JEPA Loss

The base loss is latent regression:

$$
\mathcal{L}_{\text{JEPA}} = \|\hat{z}_i - \text{sg}(z_i)\|_2^2
$$

where $\text{sg}(\cdot)$ = stop-gradient (target encoder is EMA).

#### Collapse Prevention

If you only do MSE, the model can collapse to constants. JEPA prevents this via:

1. **Variance/covariance regularization** (VICReg-style)
   - Encourage embeddings to have non-trivial variance across batch
   - Discourage correlated dimensions

2. **Multi-view consistency**
   - Create two augmented views of the same cell (gene dropout, noise)
   - Encode both, force consistency

3. **Predict multiple targets**
   - Predict both $z$ and low-dimensional biological summaries (pathway scores, PCs)

**Full loss:**

$$
\mathcal{L} = \mathcal{L}_{\text{JEPA}} + \lambda_1 \mathcal{L}_{\text{var}} + \lambda_2 \mathcal{L}_{\text{cov}}
$$

### Masking Strategy

JEPA benefits from masking because it forces reasoning about missing parts.

For expression vectors, "masking" can mean:
- Randomly zero out a subset of genes (with mask indicator)
- Drop whole gene modules/pathways
- More aggressive count downsampling

This gives:
- Robustness to dropout
- Better generalization across datasets
- Natural self-supervised signal even without perturbation labels

### Training Recipe

**Phase 1: Self-supervised pretraining**

- Ignore perturbation labels initially
- Train JEPA with masking/augmentations
- Goal: Learn a good cell state manifold

**Phase 2: Perturbation-conditioned JEPA**

- Add perturbation tokens
- Predict target embedding from baseline-ish context + perturbation

This mirrors how image/video joint models benefit from strong image priors first.

### Using the Trained Model

Once trained, you can:

1. Take a control-like context embedding $h$ (cell type + covariates)
2. Plug in perturbation $p$
3. Produce $\hat{z}(p)$ — latent representing the perturbed state

**Downstream options:**

**Stay in latent space** (preferred JEPA style):
- Nearest-neighbor retrieval of real cells
- Pathway/regulator prediction heads trained on $z$
- Differential latent shift: $\Delta z = \hat{z}(p) - \hat{z}(\varnothing)$

**Add lightweight decoder** (if needed):
- Predict expression changes $\Delta x$
- Predict sparse gene program activation

JEPA doesn't forbid decoding—it just refuses to make it the *main learning signal*.

### Evaluation Metrics

Concrete evaluations to avoid vibes-based science:

| Metric | What It Tests |
|--------|---------------|
| **Perturbation classification** from $z$ | Does embedding separate perturbations? |
| **Held-out perturbation prediction** | Generalization to unseen perturbations |
| **Combinatorial generalization** | Predict double-KO from singles |
| **DEG recovery** | Do predicted changes match observed DE genes? |
| **Pathway consistency** | Does $\Delta z$ align with known biology? |

---

## Part 6: One Important Distinction

### JEPA vs. Diffusion/Flow: How Uncertainty is Handled

| Approach | Uncertainty Handling |
|----------|---------------------|
| **Diffusion/Flow** | Explicit stochasticity, full generative modeling |
| **JEPA** | Implicit uncertainty, representation-level prediction |

But this is not a contradiction.

A natural **2025-2026 hybrid** combines:
- JEPA-style latent prediction
- Stochastic heads or flow-matching *in latent space*

This gives:
- Uncertainty without pixel-level noise
- Dynamics without autoregressive collapse

---

## Part 7: When NOT to Use Joint Latent Spaces

Joint latent spaces are powerful but not universally appropriate.

**Counter-cases:**

1. **Modalities with incompatible scales**
   - If one modality has 100× more samples, joint training may harm the smaller modality
   - Solution: Careful loss weighting or staged training

2. **Fundamentally different semantics**
   - If "distance" means different things in each modality
   - Example: Spatial vs. functional similarity in genes

3. **Conflicting optimization landscapes**
   - Different learning rates or architectures may be needed
   - Solution: Modality-specific encoders with shared predictor

4. **Privacy/data constraints**
   - If modalities can't be co-located for training

---

## Summary

### Key Takeaways

1. **Joint latent spaces** allow different data modalities to train each other, sharing priors between static and dynamic observations.

2. **JEPA** predicts embeddings not pixels, focusing on semantics while ignoring observation noise—perfect for biology where reconstruction is rarely the goal.

3. **Patch n' Pack** respects heterogeneity (variable lengths, sizes) instead of padding it away—directly applicable to genes, transcripts, and cells.

4. **Flow-based approaches** may be preferable to diffusion for biology, where "noise semantics" are unclear.

5. **For Perturb-seq and similar tasks**, JEPA provides a natural framework: predict the latent state of perturbed cells from baseline context + perturbation tokens.

### The Unifying Insight

> If two data types differ only by dimensionality or observation density, they probably want the same latent space.

Nature never padded anything to a fixed shape—engineers did. These architectures are beginning to respect that reality.

---

## References

### Goku Model
- ByteDance & HKU (2024). Goku: Native Joint Image-Video Generation

### JEPA Family
- Assran et al. (2023). [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)](https://arxiv.org/abs/2301.08243). CVPR 2023.
- Bardes et al. (2024). [V-JEPA: Video Joint Embedding Predictive Architecture](https://ai.meta.com/research/publications/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- Meta AI (2025). [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning](https://arxiv.org/abs/2506.09985)
- (2025). [VL-JEPA: Joint Embedding Predictive Architecture for Vision-language](https://arxiv.org/abs/2512.10942)

### Related Concepts
- VICReg: Variance-Invariance-Covariance Regularization
- Rectified Flow: Optimal transport for generative modeling
- Patch n' Pack: Efficient variable-length batching for Transformers

---

## Next Steps

From here, you could:
1. **Implement a minimal JEPA** for a toy biological dataset (e.g., perturbed gene expression)
2. **Explore hierarchical JEPA** for multi-scale data (patches → regions → whole samples)
3. **Compare JEPA vs. diffusion** specifically for biological counterfactual prediction
4. **Benchmark on Perturb-seq**: Norman et al. (2019) dataset is a good starting point

The interesting part isn't copying any particular model—it's recognizing the **pattern** these architectures expose and applying it to biology.
