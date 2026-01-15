# Open Research: Tokenization in Diffusion Transformers

**Status**: Active research area (as of January 2026)

**Core Question**: What is the "right" way to tokenize complex objects (images, gene expression, molecules) for transformer-based generative models?

---

## The Problem

Transformers operate on **sequences of tokens**. For natural language and DNA/RNA, tokenization is natural — these are inherently sequential. But for other modalities, tokenization feels **contrived and arbitrary**.

### The Uncomfortable Truth

**Patch-based tokenization (16×16, 8×8, etc.) is a pragmatic hack that works, but lacks principled justification.**

```
Engineering Reality: "Does it work?" ✅
Theoretical Satisfaction: "Is it right?" ❌
```

This document explores why current approaches feel unsatisfying and outlines open research directions.

---

## 1. Images: The Patch Problem

### Current Approach

**Standard practice** (ViT, DiT, Stable Diffusion 3):
```python
# Split image into fixed-size patches
patches = image.unfold(dimension=2, size=16, step=16)  # 16×16 patches
tokens = embed(patches)
output = transformer(tokens)
```

### Why This Feels Wrong

**Q1: Why 16×16?**

- Not "right" — just empirically tuned
- Different models use different sizes (2×2, 4×4, 8×8, 14×14, 16×16)
- No principled way to choose

**Q2: Should patch size depend on content?**

- Medical images (smooth gradients): Large patches OK
- Text images (fine details): Small patches needed
- Satellite images: Depends on scale of features
- **Current approach**: One size for all!

**Q3: Do patches respect semantic boundaries?**

- A 16×16 patch might contain:
  - Half a face, half background
  - Part of an object, part of another
  - Arbitrary image regions
- **Our visual cortex doesn't work this way**

### Trade-offs

| Patch Size | Pros | Cons |
|------------|------|------|
| **Small (2×2, 4×4)** | Fine details, local structure | More tokens, O(n²) attention cost |
| **Large (16×16, 32×32)** | Fewer tokens, faster | Loss of detail, coarse representation |

**The problem**: This is a hyperparameter, not a principled design choice.

---

## 2. Gene Expression: Even Less Obvious

Gene expression vectors: $x \in \mathbb{R}^{20000}$ (20K genes)

**Properties**:

- **Unordered**: No natural sequence (unlike DNA)
- **Dense**: Most genes have non-zero expression
- **Compositional**: Relative values matter
- **High-dimensional**: 10K-30K genes typical

### Current Approaches (2023-2026)

**Approach 1: Rank by Expression (Geneformer)**
```python
# Sort genes by expression level
genes_sorted = sort_by_expression(gene_expression)
tokens = [gene_1, gene_2, ..., gene_20000]
output = transformer(tokens)
```

**Problems**:

- Ranking is **arbitrary** — not biological
- 20K tokens = huge sequences (O(n²) = 400M operations)
- What about genes with same expression?
- Loses biological structure

**Approach 2: Gene Modules/Pathways**
```python
# Group genes by function
modules = {
    "glycolysis": [gene_1, gene_5, ...],
    "cell_cycle": [gene_2, gene_15, ...],
}
tokens = [module_embeddings]  # ~500 pathways
```

**Problems**:

- How to define modules? (Also arbitrary!)
- Loses individual gene information
- Ignores within-module correlations

**Approach 3: No Explicit Tokenization**
```python
# Direct embedding to latent space
z = encoder(gene_expression)  # (20000,) → (512,)
output = model(z)  # No tokens!
```

**Problems**:

- Less interpretable
- Loses biological structure
- Black box

**Approach 4: Graph-Structured (GRN-aware)**
```python
# Use gene regulatory network
grn = load_gene_regulatory_network()
output = graph_transformer(gene_expression, grn)
```

**Problems**:

- GRN knowledge incomplete
- Still 20K nodes to handle
- Which GRN to use?

### The Core Issue

**There is no natural "tokenization" for gene expression.**

Unlike images (spatial structure) or language (sequential structure), gene expression is:
- A **set** (unordered)
- A **vector** (continuous)
- A **network** (interconnected)

**Forcing it into a sequence feels wrong because it is wrong.**

---

## 3. Why Do Patches Work Despite Being Arbitrary?

### Pragmatic Reasons

**1. Computational Efficiency**
```
256×256 image = 65,536 pixels
With 16×16 patches = 256 tokens
Attention: 65,536² → 256² (65,000× reduction!)
```

**2. Transfer from NLP**

- Transformers proven for sequences
- Patches make images "sequence-like"
- Can reuse architectures

**3. Good Enough in Practice**

- ImageNet SOTA achieved
- Stable Diffusion works
- Empirical success

**4. Implementation Simplicity**

- Easy to code
- GPU-efficient
- Standard operations

### But This Doesn't Make It "Right"

**Engineering success ≠ Principled design**

The field has optimized for what works, not what makes sense.

---

## 4. Alternative Approaches (Research Frontiers)

### 4.1 Hierarchical Tokenization

**Idea**: Learn local semantics first, then group into "super tokens"

**Swin Transformer** (2021):
```
Image → Small patches → Local attention → Merge
              ↓
      "Super tokens" (hierarchical)
              ↓
      Global attention
```

**Status**: Works well, but still uses fixed patch sizes at each level.

### 4.2 Learned Tokenization

**Idea**: Don't fix patch size — learn how to tokenize!

**BEiT, VQGAN, MaskGIT**:
```python
# Instead of fixed patches
tokens = split_into_patches(image, size=16)  # Fixed

# Learn tokenization
tokens = learned_tokenizer(image)  # Adaptive!
```

**Advantages**:

- Content-aware
- Can adapt to different regions
- Potentially more semantic

**Challenges**:

- How to train the tokenizer?
- Discrete vs continuous tokens?
- Computational cost

### 4.3 Convolutional Stem

**Idea**: Use CNNs for local features, Transformers for global

```python
class HybridModel(nn.Module):
    def __init__(self):
        # CNN extracts local semantics
        self.conv_stem = ResNet(...)
        # Transformer on CNN features
        self.transformer = Transformer(...)
```

**Status**: Used in some models, but not standard for DiT.

### 4.4 No Tokenization (Continuous)

**Idea**: Work directly in continuous space

**For images**:

- Latent diffusion (VAE → continuous latent → diffusion)
- No explicit tokens

**For gene expression**:

- Direct MLP/attention on expression vector
- Treat as continuous state, not sequence

**Advantage**: No arbitrary discretization

**Disadvantage**: May lose interpretability

---

## 5. Biological Inspiration: How Should We Think About This?

### How Visual Cortex Works

```
Retina → V1 (edges) → V2 (motion) → V3 (shape) → V4 → IT (objects)
```

**Key properties**:
1. **Hierarchical**: Simple → complex features
2. **Local receptive fields** that grow
3. **Specialization**: Different areas for different features
4. **Sparse coding**: Neurons fire selectively
5. **Feedback**: Top-down and bottom-up

### Current Models vs Biology

| Aspect | Biology | Patch-based DiT |
|--------|---------|-----------------|
| **Hierarchy** | Yes (V1→V2→V3→V4) | Flat (all patches equal) |
| **Local first** | Yes (small receptive fields) | No (global attention) |
| **Adaptive** | Yes (attention, feedback) | No (fixed patches) |
| **Sparse** | Yes (selective firing) | No (dense attention) |

**Conclusion**: Current approaches are **not biologically inspired**.

### Should We Care?

**Two perspectives**:

**Pragmatic**: "Biology is slow, backprop works, patches work — who cares?"
- Valid for engineering
- Gets SOTA results

**Principled**: "Understanding biology might lead to better architectures"
- Valid for research
- May unlock new capabilities

**Reality**: Field is mostly pragmatic (for now).

---

## 6. Open Research Questions

### For Images

**Q1**: What is the optimal tokenization strategy?
- Fixed patches? Learned? Hierarchical?
- Content-adaptive?
- Task-specific?

**Q2**: Can we learn tokenization end-to-end?
- Jointly with the generative model?
- Discrete vs continuous?

**Q3**: How important is biological plausibility?
- Should we model V1→V2→V3→V4?
- Or is attention enough?

### For Gene Expression

**Q4**: What is the "right" representation?
- Tokens (if so, what kind)?
- Continuous embeddings?
- Graph structure?

**Q5**: Should tokenization respect biological structure?
- Gene modules/pathways?
- Regulatory networks?
- Or learn from data?

**Q6**: How to handle high dimensionality?
- 20K genes → how many tokens?
- Latent space diffusion?
- Hierarchical representation?

### General Questions

**Q7**: Is tokenization necessary at all?
- Can we do generative modeling without tokens?
- Continuous-space alternatives?

**Q8**: Should tokenization be modality-specific?
- Images: Patches
- Audio: Time patches
- Gene expression: ???
- Or unified approach?

**Q9**: How to evaluate tokenization quality?
- Reconstruction error?
- Downstream task performance?
- Interpretability?

---

## 7. Current State of the Field (January 2026)

### What's Working

**For images**:

- Fixed patches (8×8, 16×16) are standard
- Empirically tuned per model
- Stable Diffusion 3, Sora use patch-based approaches

**For gene expression**:

- Multiple approaches being explored
- No clear winner yet
- Geneformer (ranking), scPPDM (tabular), others

### What's Being Researched

**Active areas**:
1. Learned tokenization (VQ-VAE, MaskGIT)
2. Hierarchical models (Swin, PVT)
3. Hybrid CNN-Transformer
4. Graph-structured attention
5. Continuous-space alternatives

### What's Still Unknown

**Open problems**:

- Principled way to choose patch size
- Optimal tokenization for non-image modalities
- Whether biological inspiration helps
- Unified tokenization across modalities

---

## 8. Recommendations for Practitioners

### For Image Generation (DiT)

**Current best practice**:
```python
# Use empirically-tuned patch sizes
patch_size = 8  # For 256×256 images (DiT-XL/8)
# or
patch_size = 4  # For higher quality (more compute)
```

**Experiment with**:

- Different patch sizes for your data
- Hierarchical approaches if quality matters
- Latent diffusion (VAE + diffusion) to avoid tokenization

### For Gene Expression

**Recommended approach** (as of 2026):
```python
# Option 1: No explicit tokenization
z = encoder(gene_expression)  # (20000,) → (512,)
output = diffusion_model(z)

# Option 2: Biologically-structured
grn = load_gene_regulatory_network()
output = graph_diffusion(gene_expression, grn)

# Option 3: Learned modules
modules = learn_gene_modules(data)  # Data-driven
tokens = embed_by_modules(gene_expression, modules)
output = transformer(tokens)
```

**Then**:

- Compare approaches empirically
- Publish ablation study
- Let performance guide you

### General Advice

**Start simple**:
1. Use standard approaches (patches for images, embeddings for other)
2. Get baseline working
3. Then experiment with alternatives

**Don't overthink**:

- If patches work for your task, use them
- Principled design is nice, but results matter

**But do explore**:

- This is an open research area
- Novel tokenization strategies could be publishable
- Especially for non-image modalities

---

## 9. Future Directions

### Near-term (2026-2027)

**Likely developments**:
1. More learned tokenization methods
2. Better hierarchical models
3. Modality-specific tokenization strategies
4. Improved understanding of why patches work

### Medium-term (2027-2029)

**Possible breakthroughs**:
1. Unified tokenization framework
2. Biologically-inspired alternatives that match SOTA
3. Continuous-space generative models (no tokens)
4. Neural architecture search for tokenization

### Long-term (2029+)

**Speculative**:
1. Fundamental rethinking of tokenization
2. New architectures that don't need tokens
3. True biological plausibility
4. Modality-agnostic generative models

---

## 10. The Bigger Picture

### The Tension

```
Engineering Pragmatism     vs     Principled Design
"Does it work?"            vs     "Is it right?"
Empirical tuning           vs     Theory-driven
Fast iteration             vs     Deep understanding
```

**Current state**: Pragmatism dominates
- Patch sizes: Empirically tuned
- Architecture choices: What works on benchmarks
- Limited theoretical understanding

**Future direction**: More principled approaches
- Understanding WHY things work
- Biologically-inspired designs
- Learned, adaptive strategies

### Why This Matters

**For science**:

- Understanding principles leads to better models
- Biological inspiration may unlock new capabilities
- Theory guides experimentation

**For engineering**:

- Principled designs generalize better
- Less hyperparameter tuning
- More robust to distribution shift

**For biology applications**:

- Gene expression needs better representations
- Biological structure should inform design
- Interpretability matters

---

## 11. Conclusion

**The honest assessment**:

**Patch-based tokenization is arbitrary and unnatural.**

- 16×16 is not "right" — it's empirically tuned
- Should vary by resolution, task, data
- Doesn't respect semantic boundaries
- Not biologically inspired

**But it works.**

- Achieves SOTA on many tasks
- Computationally efficient
- Easy to implement

**For gene expression, it's even worse.**

- No natural tokenization exists
- Current approaches are hacks
- Open research problem

**The field is still figuring this out.**

- Active research area
- No consensus
- Your skepticism is warranted

**Recommendations**:
1. Use standard approaches to get started
2. Experiment with alternatives
3. Let performance guide you
4. Contribute to the research!

---

## References

### Tokenization Approaches

**Patch-based**:

- Dosovitskiy et al. (2020): "An Image is Worth 16x16 Words" (ViT)
- Peebles & Xie (2023): "Scalable Diffusion Models with Transformers" (DiT)

**Hierarchical**:

- Liu et al. (2021): "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- Wang et al. (2021): "Pyramid Vision Transformer"

**Learned Tokenization**:

- Bao et al. (2021): "BEiT: BERT Pre-Training of Image Transformers"
- Esser et al. (2021): "Taming Transformers for High-Resolution Image Synthesis" (VQGAN)
- Chang et al. (2022): "MaskGIT: Masked Generative Image Transformer"

**Gene Expression**:

- Theodoris et al. (2023): "Transfer learning enables predictions in network biology" (Geneformer)
- Cui et al. (2024): "scGPT: Toward Building a Foundation Model for Single-Cell Multi-omics"

### Biological Inspiration

- Hinton et al. (2017): "Dynamic Routing Between Capsules" (Capsule Networks)
- Rao & Ballard (1999): "Predictive coding in the visual cortex"

---

## Discussion Questions

**For researchers**:
1. Can we develop a principled theory of tokenization?
2. Should tokenization be learned end-to-end with the model?
3. How important is biological plausibility?
4. Can we unify tokenization across modalities?

**For practitioners**:
1. How to choose patch size for my data?
2. When should I use hierarchical models?
3. Is learned tokenization worth the complexity?
4. How to tokenize gene expression data?

**For the field**:
1. Are we over-engineering tokenization?
2. Should we move beyond tokens entirely?
3. What can biology teach us?
4. How to balance pragmatism and principles?

---

**Status**: Open research area — contribute your ideas!

**Last updated**: January 13, 2026
