# Understanding Transformer Data Shapes

A comprehensive guide to tensor shapes in transformers, from input to output, including parameter-efficient fine-tuning methods like LoRA.

**Key question**: When you see `output = transformer(tokens)`, what exactly is happening to the data shapes?

---

## Overview

This guide demystifies the transformer abstraction by explaining:

1. **Input/output contracts** — What transformers actually take and return
2. **Internal transformations** — Shape changes inside attention and feedforward layers
3. **Layer stacking** — How shapes flow through multiple blocks
4. **Output interpretation** — What the hidden states mean
5. **LoRA and adapters** — How fine-tuning methods affect shapes
6. **Practical implications** — Design principles for biological models

---

## 1. The Core Contract: Shape-Preserving Transformation

### What Transformers Actually Do

At the highest level, a transformer performs **one operation**:

> It maps a *sequence of vectors* to another *sequence of vectors of the same length*.

**Mathematical signature**:

$$
\text{Transformer}: \mathbb{R}^{B \times T \times d_{\text{model}}} \longrightarrow \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

Where:
- $B$ = batch size
- $T$ = number of tokens (sequence length)
- $d_{\text{model}}$ = embedding/hidden dimension

### Key Insight

**Transformers are shape-preserving** in both token dimension and feature dimension.

They do **NOT**:
- Reduce token count
- Change embedding size
- Pool by default

They **rewrite representations**, not compress them.

### Example

```python
# Input
tokens = torch.randn(32, 64, 512)  # (batch=32, tokens=64, dim=512)

# Transformer
output = transformer(tokens)

# Output (same shape!)
print(output.shape)  # torch.Size([32, 64, 512])
```

---

## 2. What Is a "Token"?

### Definition

A token is simply a vector in $\mathbb{R}^{d_{\text{model}}}$.

How you obtained it is **upstream business**:
- **Words** → embeddings (NLP)
- **Image patches** → linear projection (Vision)
- **Gene expression** → encoder output (Biology)
- **Latent codes** → diffusion latents (Generative models)

By the time it reaches the transformer, **the transformer doesn't care about the origin**.

### The Contract

```python
tokens: [B, T, d_model]
```

This is the **interface contract**.

Everything else—biology, language, pixels—is already baked into those vectors.

### Example: Different Modalities, Same Shape

```python
# Text (BERT)
text_tokens = word_embeddings(text)  # (32, 128, 768)

# Images (ViT)
image_tokens = patch_embeddings(image)  # (32, 196, 768)

# Gene expression (custom)
gene_tokens = gene_encoder(expression)  # (32, 64, 768)

# All can use the same transformer!
transformer = Transformer(d_model=768)
output = transformer(tokens)  # Works for all modalities
```

---

## 3. Inside a Transformer Block

A standard transformer block has two sublayers:

1. **Multi-head self-attention**
2. **Position-wise feedforward network (MLP)**

Both obey the same rule:

> Input shape = Output shape = $[B, T, d_{\text{model}}]$

Let's examine each.

---

## 3.1 Self-Attention: Where Tokens Communicate

### Input

$$
X \in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

### Step 1: Linear Projections

Three learned linear maps create queries, keys, and values:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

Each still has shape:

$$
[B, T, d_{\text{model}}]
$$

### Step 2: Reshape into Heads

$$
[B, h, T, d_{\text{head}}] \quad \text{where} \quad d_{\text{head}} = d_{\text{model}} / h
$$

Where $h$ is the number of attention heads.

### Step 3: Attention Scores (The Key Transformation)

This is the **only moment** where shape meaningfully changes:

$$
\text{Attention scores: } QK^\top \Rightarrow [B, h, T, T]
$$

This is the "who attends to whom" matrix.

**Important**: This $T \times T$ matrix is **temporary** and never leaves the block.

### Step 4: Apply Attention and Recombine

After softmax and weighting with values:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}}\right)V
$$

Everything collapses back to:

$$
[B, T, d_{\text{model}}]
$$

### Visualization

```python
# Input
X = (B, T, d_model)

# Project to Q, K, V
Q = X @ W_Q  # (B, T, d_model)
K = X @ W_K  # (B, T, d_model)
V = X @ W_V  # (B, T, d_model)

# Reshape to heads
Q = Q.view(B, T, h, d_head).transpose(1, 2)  # (B, h, T, d_head)
K = K.view(B, T, h, d_head).transpose(1, 2)  # (B, h, T, d_head)
V = V.view(B, T, h, d_head).transpose(1, 2)  # (B, h, T, d_head)

# Attention scores (TEMPORARY SHAPE CHANGE)
scores = Q @ K.transpose(-2, -1)  # (B, h, T, T) ← Token-token interaction
attn = softmax(scores / sqrt(d_head))

# Apply attention
out = attn @ V  # (B, h, T, d_head)

# Recombine heads
out = out.transpose(1, 2).contiguous().view(B, T, d_model)  # (B, T, d_model)
```

### Key Takeaway

Attention temporarily creates a **token-token interaction matrix** $(T \times T)$, but the output is always $(B, T, d_{\text{model}})$.

---

## 3.2 Feedforward Network: No Token Mixing

The MLP is applied **independently to each token**:

$$
\text{FFN}(x_t) = W_2 \sigma(W_1 x_t + b_1) + b_2
$$

### Shapes

$$
[B, T, d_{\text{model}}] \rightarrow [B, T, d_{\text{ff}}] \rightarrow [B, T, d_{\text{model}}]
$$

Where $d_{\text{ff}}$ is typically $4 \times d_{\text{model}}$.

### Implementation

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x: (B, T, d_model)
        x = self.linear1(x)  # (B, T, d_ff)
        x = self.activation(x)
        x = self.linear2(x)  # (B, T, d_model)
        return x
```

### Key Takeaway

**No cross-token interaction** in the feedforward layer. All mixing happens in attention.

---

## 3.3 Residuals and Normalization

Residual connections ensure:

$$
\text{output} = X + \text{sublayer}(X)
$$

This is why the shape **must** stay the same.

### Complete Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, T, d_model)
        
        # Attention with residual
        x = x + self.attention(self.norm1(x))  # (B, T, d_model)
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))  # (B, T, d_model)
        
        return x  # (B, T, d_model)
```

**Transformers are iterative representation refiners.**

---

## 4. Stacking Layers: Still the Same Shape

A transformer with $L$ layers is just:

$$
X^{(0)} \rightarrow X^{(1)} \rightarrow \cdots \rightarrow X^{(L)}
$$

Each $X^{(\ell)} \in \mathbb{R}^{B \times T \times d_{\text{model}}}$.

### Visualization

```python
# Input
X_0 = tokens  # (B, T, d_model)

# Layer 1
X_1 = block_1(X_0)  # (B, T, d_model)

# Layer 2
X_2 = block_2(X_1)  # (B, T, d_model)

# ...

# Layer L
X_L = block_L(X_{L-1})  # (B, T, d_model)

# Output
output = X_L  # (B, T, d_model)
```

### Interpretation

```python
output = transformer(tokens)
```

means:

> "Each token has been rewritten $L$ times using global context."

Nothing more. Nothing less.

---

## 5. What Is the Output?

### The Raw Output

The transformer itself outputs:

$$
\text{hidden states} \in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

### What It Means Depends on the Task

The transformer doesn't decide what the output means—**the head does**.

| Task | Head Operation | Output Shape |
|------|----------------|--------------|
| **Language modeling** | Each token predicts next token | $(B, T, \text{vocab\_size})$ |
| **Classification** | Pool or select CLS token | $(B, \text{num\_classes})$ |
| **Diffusion** | Each token predicts noise/velocity | $(B, T, d_{\text{model}})$ |
| **Gene expression** | Each token predicts distribution params | $(B, T, \text{num\_genes})$ |

### Example: Different Heads

```python
# Transformer output (same for all tasks)
hidden = transformer(tokens)  # (B, T, d_model)

# Language modeling head
logits = lm_head(hidden)  # (B, T, vocab_size)

# Classification head (pool first)
pooled = hidden[:, 0, :]  # (B, d_model) - CLS token
logits = classifier(pooled)  # (B, num_classes)

# Diffusion head
noise_pred = diffusion_head(hidden)  # (B, T, d_model)

# Gene expression head
gene_params = gene_head(hidden)  # (B, T, num_genes)
```

---

## 6. Pooling Is Not Part of the Transformer

If you see operations like:

- **CLS token** — Select first token
- **Mean pooling** — Average over tokens
- **Attention pooling** — Weighted average

These are **post-transformer operations**, not transformer logic.

### Example: Pooling Operations

```python
# Transformer output
hidden = transformer(tokens)  # (B, T, d_model)

# CLS token (BERT-style)
cls_output = hidden[:, 0, :]  # (B, d_model)

# Mean pooling
mean_output = hidden.mean(dim=1)  # (B, d_model)

# Attention pooling
attn_weights = attention_pooling(hidden)  # (B, T, 1)
pooled_output = (hidden * attn_weights).sum(dim=1)  # (B, d_model)
```

### Why This Matters for Biology

**Token-level outputs ≠ Sample-level outputs**

In gene expression models:
- **Token-level**: Each token represents a gene module/pathway
- **Sample-level**: Need to aggregate tokens for cell-level prediction

Design choice: How to aggregate?

---

## 7. LoRA and Adapters: What Changes?

### Short Answer

**Nothing about the output shape changes.**

### Long Answer

The *function* changes, not the *type signature*.

---

## 7.1 LoRA in Detail

### Concept

LoRA replaces a weight matrix $W$ with:

$$
W_{\text{eff}} = W + \Delta W \quad \text{where} \quad \Delta W = AB
$$

- $A \in \mathbb{R}^{d_{\text{out}} \times r}$
- $B \in \mathbb{R}^{r \times d_{\text{in}}}$
- $r \ll d_{\text{model}}$ (typically 4-16)

### Key Point

$$
W_{\text{eff}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
$$

**Same shape as before.**

### Implementation

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        
        # Original weight (frozen)
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        
        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original output
        output = self.linear(x)
        
        # LoRA update
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        return output + lora_output
```

### Shape Analysis

```python
# Input
x = (B, T, d_in)

# Original linear
W: (d_out, d_in)
output = x @ W.T  # (B, T, d_out)

# LoRA
A: (d_in, r)
B: (r, d_out)
lora_output = x @ A @ B  # (B, T, d_out)

# Combined
final_output = output + lora_output  # (B, T, d_out)
```

**Input/output shapes are identical** to the original layer.

---

## 7.2 Why This Is Important

### Functional Perturbation

LoRA is a *functional perturbation* of the model:
- Bends attention geometry
- Nudges feature subspaces
- Steers behavior

But it **does not change the interface**.

### Practical Benefits

This enables:

```python
# Swap LoRA modules per task
model.load_lora("task_A.pt")
output_A = model(input)

model.load_lora("task_B.pt")
output_B = model(input)

# Compose multiple skills
model.load_lora(["skill_1.pt", "skill_2.pt"])
output = model(input)
```

**No downstream code changes needed.**

### Software Engineering Gold

From a software perspective, this is powerful:
- **Modular** — Swap adapters without touching backbone
- **Composable** — Combine multiple LoRA modules
- **Efficient** — Store multiple task-specific models cheaply

---

## 8. Thinking in Type Signatures

### Mental Model

Think of a transformer as having a **type**:

$$
\text{Transformer}[T, d]: \text{Tokens}[T, d] \rightarrow \text{Tokens}[T, d]
$$

**LoRA, adapters, fine-tuning, freezing—none of these change the type.**

### What Changes Types?

Only **encoders, decoders, and heads** change types:

```python
# Encoder: Raw data → Tokens
encoder: counts (num_genes) → tokens (T, d)

# Transformer: Tokens → Tokens
transformer: tokens (T, d) → tokens (T, d)

# Decoder: Tokens → Output
decoder: tokens (T, d) → distributions (num_genes)
```

### Example: Complete Pipeline

```python
# Gene expression → Latent tokens
tokens = encoder(gene_expression)  # (20000,) → (64, 256)

# Latent diffusion (with LoRA)
denoised = diffusion_transformer(tokens, t)  # (64, 256) → (64, 256)

# Tokens → Gene expression parameters
params = decoder(denoised)  # (64, 256) → (20000,)
```

**Type changes happen at boundaries, not inside the transformer.**

---

## 9. Implications for Biology Models

### Key Insight

> Once gene expression is mapped into a token space, **all foundation-model machinery becomes legal.**

Diffusion, DiT, CFG, LoRA, adapters—they all operate on:

$$
[B, T, d]
$$

### Design Freedom

Your real design choices are:

1. **How you tokenize biology** — Encoder design
2. **How you decode outputs** — Decoder design
3. **How you condition transformations** — Conditioning mechanisms

**The transformer itself is just the universal mixer.**

### Example: Modular Design

```python
# Tokenization choice
encoder = LatentTokenEncoder(num_genes=20000, num_tokens=64, token_dim=256)
# OR
encoder = PathwayTokenEncoder(pathways=msigdb_hallmark, token_dim=256)

# Transformer (same for both!)
transformer = DiT(embed_dim=256, depth=12, num_heads=8)

# Add LoRA (same interface!)
transformer = LoRA.wrap(transformer, rank=8)

# Decoder choice
decoder = NegativeBinomialDecoder(token_dim=256, num_genes=20000)
# OR
decoder = ZINBDecoder(token_dim=256, num_genes=20000)
```

**Mix and match without changing the transformer.**

---

## 10. Complete Shape Trace: End-to-End Example

### Task: scRNA-seq Latent Diffusion

```python
# ═══════════════════════════════════════════════════════════
# INPUT: Gene expression counts
# ═══════════════════════════════════════════════════════════
gene_expression = (batch=32, num_genes=20000)

# ═══════════════════════════════════════════════════════════
# ENCODER: Counts → Latent tokens
# ═══════════════════════════════════════════════════════════
latent_tokens = encoder(gene_expression)
# Shape: (32, 64, 256)
#        ↑   ↑   ↑
#        |   |   └─ Token dimension
#        |   └───── Number of tokens
#        └───────── Batch size

# ═══════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════
pos_embed = (1, 64, 256)  # Broadcasts across batch
latent_tokens = latent_tokens + pos_embed
# Shape: (32, 64, 256)

# ═══════════════════════════════════════════════════════════
# CONDITIONING: Perturbation embedding
# ═══════════════════════════════════════════════════════════
perturbation = (32, 128)  # Perturbation embedding
condition = condition_proj(perturbation)  # (32, 256)

# ═══════════════════════════════════════════════════════════
# DIFFUSION TRANSFORMER (with LoRA)
# ═══════════════════════════════════════════════════════════
# Add noise
t = torch.randint(0, 1000, (32,))
noise = torch.randn(32, 64, 256)
noisy_tokens = sqrt(alpha_t) * latent_tokens + sqrt(1 - alpha_t) * noise
# Shape: (32, 64, 256)

# Transformer forward (SHAPE PRESERVED!)
for block in transformer.blocks:
    # Self-attention
    attn_out = block.attention(noisy_tokens)  # (32, 64, 256)
    noisy_tokens = noisy_tokens + attn_out
    
    # Conditioning (FiLM)
    scale = film.scale_net(condition)  # (32, 256)
    shift = film.shift_net(condition)  # (32, 256)
    noisy_tokens = noisy_tokens * scale.unsqueeze(1) + shift.unsqueeze(1)
    
    # FFN
    ffn_out = block.ffn(noisy_tokens)  # (32, 64, 256)
    noisy_tokens = noisy_tokens + ffn_out

denoised_tokens = noisy_tokens  # (32, 64, 256)

# ═══════════════════════════════════════════════════════════
# DECODER: Latent tokens → Gene expression parameters
# ═══════════════════════════════════════════════════════════
gene_params = decoder(denoised_tokens)
# Shape: (32, 20000, 2)
#        ↑   ↑       ↑
#        |   |       └─ Parameters (mean, dispersion)
#        |   └───────── Genes
#        └───────────── Batch size

# ═══════════════════════════════════════════════════════════
# OUTPUT: NB/ZINB distribution
# ═══════════════════════════════════════════════════════════
mean, dispersion = gene_params.chunk(2, dim=-1)
# mean: (32, 20000)
# dispersion: (32, 20000)
```

### Shape Summary Table

| Stage | Operation | Input Shape | Output Shape |
|-------|-----------|-------------|--------------|
| **Input** | Raw counts | `(32, 20000)` | - |
| **Encoder** | Compress to tokens | `(32, 20000)` | `(32, 64, 256)` |
| **Pos Embed** | Add position info | `(32, 64, 256)` | `(32, 64, 256)` |
| **Diffusion** | Add noise | `(32, 64, 256)` | `(32, 64, 256)` |
| **Transformer** | Denoise (12 blocks) | `(32, 64, 256)` | `(32, 64, 256)` |
| **Decoder** | Tokens → params | `(32, 64, 256)` | `(32, 20000, 2)` |
| **Output** | NB distribution | `(32, 20000, 2)` | - |

**Key observation**: Transformer operates entirely in `(B, T, d)` space.

---

## Key Takeaways

### Core Principles

1. **Transformers preserve shape** — Input and output have same dimensions
2. **Tokens are the contract** — `(B, T, d_model)` is the universal interface
3. **Attention creates temporary interactions** — `(T, T)` matrix never leaves the block
4. **LoRA doesn't change shapes** — Only changes the function, not the signature
5. **Pooling is external** — Not part of the transformer itself

### Design Implications

1. **Tokenization is key** — How you enter the `(B, T, d)` space matters most
2. **Decoding is flexible** — How you exit the space is task-specific
3. **Conditioning is modular** — Can inject at multiple points
4. **Adaptation is cheap** — LoRA/adapters don't change interfaces

### Mental Model

A transformer does not generate meaning. It **redistributes information across tokens while preserving shape**.

Everything interesting happens in how you *enter* and *exit* that space.

---

## Related Documents

- [leveraging_foundation_models_v2.md](leveraging_foundation_models_v2.md) — Design patterns for adaptation
- [../latent_diffusion/](../latent_diffusion/) — Latent diffusion implementation
- [../DiT/01_dit_foundations.md](../DiT/01_dit_foundations.md) — DiT architecture details
- [../DDPM/02a_diffusion_arch_gene_expression.md](../DDPM/02a_diffusion_arch_gene_expression.md) — Tokenization for biology

---

## References

**Transformers**:

- Vaswani et al. (2017): "Attention Is All You Need"
- Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
- Dosovitskiy et al. (2021): "An Image is Worth 16x16 Words: Transformers for Image Recognition"

**Parameter-efficient fine-tuning**:

- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
- Houlsby et al. (2019): "Parameter-Efficient Transfer Learning for NLP"

**Diffusion transformers**:

- Peebles & Xie (2023): "Scalable Diffusion Models with Transformers" (DiT)
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"
