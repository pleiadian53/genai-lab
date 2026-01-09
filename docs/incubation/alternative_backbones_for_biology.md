# Alternative Backbones for Biological Generative Models

This document explores alternatives to Transformers for generative modeling in biology, where tokenization may not be natural. We focus on state-space models (SSMs), the tokenization problem for gene expression, and architectures that respect biological structure.

---

## 1. The Core Question

Modern generative models (DiT + rectified flow) use Transformers as the backbone. But Transformers require **tokenization** — representing data as a sequence of discrete units.

**For images**: Patches work well (16×16 pixels → token)

**For text**: Subwords work well (BPE, SentencePiece)

**For gene expression**: What's the natural token?

This document argues that **tokenization is optional** and explores alternatives.

---

## 2. What Diffusion/Flow Models Actually Need

Strip away the architecture. A diffusion or rectified-flow model needs:

$$
f_\theta(x_t, t, c) \rightarrow \text{vector field}
$$

**Requirements:**

- Accept a state representation
- Condition on time $t$
- Optionally condition on context $c$
- Output a vector of the same dimensionality as the state

**The real requirement:**

> A model capable of learning global dependencies and time-conditioned transformations.

Transformers satisfy this — but they are **not unique**.

---

## 3. State-Space Models as Diffusion Backbones

### Why SSMs Are Philosophically Aligned

Rectified flow and diffusion define **continuous-time dynamics**:

$$
\frac{dx}{dt} = v_\theta(x, t)
$$

State-space models are *literally designed* to model dynamics:

$$
\frac{dh}{dt} = Ah + Bx, \quad y = Ch + Dx
$$

This is not a coincidence — both frameworks think in terms of **state evolution**.

### Candidate Architectures

| Architecture | Key Feature | Biological Fit |
|--------------|-------------|----------------|
| **S4** | Structured state space | Long-range dependencies |
| **Mamba** | Selective state spaces | Input-dependent dynamics |
| **Hyena** | Implicit long convolutions | Efficient sequence modeling |
| **HyenaDNA** | DNA-specific Hyena | Genomic sequences |

### Why Transformers Won Historically

- Easy to scale (attention is parallelizable)
- Clean conditioning via cross-attention
- Unified modalities early (text, image, audio)
- Infrastructure exists (FlashAttention, etc.)

But this is **historical inertia**, not a fundamental requirement.

---

## 4. The Tokenization Problem for Gene Expression

### What Gene Expression Looks Like

$$
x \in \mathbb{R}^{G}
$$

where $G \approx 20,000$ genes.

**Properties:**

- **Unordered**: No natural sequence (genes have no inherent order)
- **Dense**: Most genes have non-zero expression
- **Compositional**: Relative, not absolute (TPM sums to 1M)
- **Population-relative**: Meaning depends on context

### The Problem with "Genes as Tokens"

Approaches like Geneformer:

1. Rank genes by expression level
2. Treat ranked genes as a sequence
3. Apply Transformer

This *works*, but feels **ontologically wrong**:

> Ranking genes is not a natural ordering of biological state — it's an engineering trick.

The ranking changes between samples, destroying any consistent "position" meaning.

### Why This Matters

If the representation is unnatural:

- The model learns spurious patterns
- Generalization suffers
- Interpretability is compromised
- Biological priors are ignored

---

## 5. Better Representations for Gene Expression

### Option A: State Vector (No Tokens)

**Idea**: Treat expression as a single state vector, not a sequence.

```
x_t ∈ ℝ^G → MLP/SSM backbone → v_θ(x_t, t) ∈ ℝ^G
```

**Implementation:**

- Backbone: MLP, SSM, or continuous-time operator
- Time-conditioning: FiLM modulation
- No tokenization at all

**Why it works:**

- Aligns with rectified flow (learning velocity in gene-expression space)
- Respects the unordered nature of genes
- Simple and honest

**Limitation**: Scales poorly with $G$ (quadratic in MLP, but SSMs help).

### Option B: Latent-Space Diffusion

**Idea**: Don't tokenize raw expression — tokenize latent representations.

```
Expression → VAE Encoder → z ∈ ℝ^d → Diffusion → z' → VAE Decoder → Expression
```

**Why it works:**

- Latent space is smooth and lower-dimensional
- No artificial ordering
- No sparsity pathologies
- VAE decoder handles count structure (NB/ZINB)

**This is where JEPA, VAEs, and diffusion naturally converge.**

See `docs/incubation/generative-ai-for-gene-expression-prediction.md` for details on count handling.

### Option C: Set-Based Representations

**Idea**: If you must use tokens, respect the unordered structure.

```
Expression → Gene embeddings × expression values → Set Transformer → Output
```

**Implementation:**

- Genes have learned embeddings
- Expression value modulates each embedding
- Use permutation-invariant operators (Set Transformer, DeepSets)
- Attention without positional encoding

**Why it works:**

- Respects symmetry (gene order doesn't matter)
- Biologically meaningful (genes are entities, not positions)

### Option D: Dynamics-First (SSM-Friendly)

**Idea**: If your data has temporal structure, make **time** the sequence axis.

```
[x(t₁), x(t₂), ..., x(tₙ)] → SSM backbone → Predicted dynamics
```

**Applicable to:**

- Time-series expression
- Perturb-seq (pre/post perturbation)
- Differentiation trajectories
- Drug response curves

**Why SSMs shine here:**

- Designed for temporal dynamics
- Efficient for long sequences
- Natural fit for biological processes

---

## 6. Proposed Architecture: Latent Rectified Flow + SSM

Combining the best ideas:

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
├─────────────────────────────────────────────────────────┤
│  Expression (counts)                                     │
│       ↓                                                  │
│  log1p + normalize                                       │
│       ↓                                                  │
│  VAE Encoder → z ∈ ℝ^d (latent state)                   │
│       ↓                                                  │
│  [z(t₁), z(t₂), ...] (if temporal) or z (if static)    │
│       ↓                                                  │
│  SSM/Mamba backbone with time modulation                 │
│       ↓                                                  │
│  Rectified flow loss: ||v_θ(z_t, t) - (z₁ - z₀)||²     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Sampling Pipeline                      │
├─────────────────────────────────────────────────────────┤
│  z(1) ~ N(0, I)                                         │
│       ↓                                                  │
│  ODE: dz/dt = -v_θ(z, t) from t=1 to t=0               │
│       ↓                                                  │
│  z(0) ≈ latent data                                     │
│       ↓                                                  │
│  VAE Decoder (NB/ZINB) → Expression (counts)            │
└─────────────────────────────────────────────────────────┘
```

**Properties:**

- No fake tokens or gene ranking
- Proper count handling via NB/ZINB decoder
- SSM models temporal dynamics naturally
- Rectified flow gives simple, stable training

---

## 7. When Tokenization IS Biologically Meaningful

Tokenization isn't always wrong. It's appropriate when:

### Pathways as Tokens

Gene sets (pathways, modules) have biological meaning:

```
Expression → Pathway activity scores → Tokens
```

Each token represents a functional unit (e.g., "cell cycle", "apoptosis").

### Cells as Tokens (Single-Cell)

In single-cell data, cells are natural units:

```
[cell₁, cell₂, ..., cellₙ] → Cell embeddings → Transformer
```

Attention between cells captures cell-cell interactions.

### Genomic Positions (DNA/RNA)

For sequence data, positions are real:

```
[A, T, G, C, ...] → Nucleotide embeddings → Transformer/Hyena
```

HyenaDNA works well here because the sequence axis is meaningful.

---

## 8. Comparison: Transformer vs SSM for Biology

| Aspect | Transformer | SSM (Mamba/S4) |
|--------|-------------|----------------|
| **Sequence assumption** | Required | Optional |
| **Temporal dynamics** | Learned implicitly | Native |
| **Long-range** | O(n²) attention | O(n) or O(n log n) |
| **Conditioning** | Cross-attention, AdaLN | FiLM, state injection |
| **Biological fit** | Good for cells, pathways | Good for trajectories |
| **Infrastructure** | Mature | Emerging |

**Recommendation:**

- **Static expression**: Latent diffusion with MLP or small Transformer
- **Temporal data**: SSM backbone
- **Cell populations**: Transformer with cells as tokens
- **Genomic sequences**: Hyena/HyenaDNA

---

## 9. Research Directions

### Near-Term (Implementable Now)

1. **Latent rectified flow for gene expression**
   - VAE with NB decoder + flow matching in latent space
   - Compare with direct diffusion on log-transformed data

2. **Set Transformer for expression**
   - Permutation-invariant architecture
   - Benchmark against Geneformer-style ranking

### Medium-Term (Requires Development)

3. **Mamba backbone for perturb-seq**
   - Temporal modeling of perturbation responses
   - Compare with Transformer on prediction tasks

4. **Hybrid architectures**
   - SSM for temporal dynamics + Transformer for cell interactions
   - Multi-scale modeling

### Long-Term (Research Questions)

5. **When does tokenization help vs hurt?**
   - Systematic comparison across biological data types
   - Theoretical analysis of inductive biases

6. **Biological priors in architecture**
   - Gene regulatory networks as attention masks
   - Pathway structure as hierarchical tokens

---

## 10. Key Takeaways

> **Tokenization is a convenience for architectures, not a requirement of the data.**

**For gene expression:**

- Avoid forcing genes into sequences
- Prefer latent-space methods
- Use set-based or state-vector representations
- Let SSMs handle temporal structure

**The organizing principle:**

```
Data structure → Representation → Architecture
     ↓                ↓              ↓
  Unordered      State vector      MLP/SSM
  Temporal       Sequence          SSM/Mamba
  Cells          Set of tokens     Set Transformer
  Genomic        True sequence     Hyena/Transformer
```

---

## References

- Gu et al. (2022) - "Efficiently Modeling Long Sequences with Structured State Spaces" (S4)
- Gu & Dao (2023) - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Nguyen et al. (2023) - "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution"
- Lee et al. (2019) - "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
- Theodoris et al. (2023) - "Transfer learning enables predictions in network biology" (Geneformer)

---

## Related Documents

- [Rectified Flow Tutorial](../flow_matching/rectifying_flow.md)
- [Diffusion Transformer Tutorial](../diffusion/DiT/diffusion_transformer.md)
- [Generative AI for Gene Expression](generative-ai-for-gene-expression-prediction.md)
- [Joint Latent Spaces and JEPA](joint_latent_space_and_JEPA.md)
