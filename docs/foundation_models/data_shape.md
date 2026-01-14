
Transformer is often abstracted in code as: 

```
output = transformer(tokens)
```

How do I conceptualize the input(s) and output(s) for a transformer? What are the key transformation of data shapes in the complex transformer layers? What are the output data shapes? What if we consider parameter efficient fine-tuning methods like LoRA, etc, then what would be the output data shape?

---

Good—this is exactly the right place to slow down and de-mythologize the incantation

```
output = transformer(tokens)
```

That line hides *a lot* of structure. Let’s unpack it carefully, but without turning it into a soup of tensor algebra.

I’ll do this in layers of abstraction, because that’s how transformers themselves are built.

---

## 1. What a transformer *really* takes as input

At the highest level, a transformer does **one thing**:

> It maps a *set or sequence of vectors* to another *set or sequence of vectors of the same length*.

So the most honest signature is:

[
\text{Transformer}: \mathbb{R}^{B \times T \times d_{\text{model}}}
;\longrightarrow;
\mathbb{R}^{B \times T \times d_{\text{model}}}
]

Where:

* (B) = batch size
* (T) = number of tokens
* (d_{\text{model}}) = embedding / hidden dimension

That’s it.
No magic yet.

If you remember only one thing: **transformers are shape-preserving in time/token dimension and feature dimension.**

They *do not*:

* reduce token count
* change embedding size
* pool by default

They **rewrite representations**, not compress them.

---

## 2. What exactly is a “token” here?

A token is just a vector in (\mathbb{R}^{d_{\text{model}}}).

How you got it is upstream business:

* words → embeddings
* image patches → linear projection
* gene expression → encoder output
* latent codes → diffusion latents

By the time it hits the transformer, the transformer does not care.

So this abstraction:

```
tokens: [B, T, d_model]
```

is the *contract*.

Everything else—biology, language, pixels—is already baked into those vectors.

---

## 3. Inside a transformer block: what changes, what doesn’t

A standard transformer block has two big sublayers:

1. **Multi-head self-attention**
2. **Position-wise feedforward network (MLP)**

Both obey the same structural rule:

> Input shape = output shape = ([B, T, d_{\text{model}}])

Let’s zoom in.

---

### 3.1 Self-attention: the only place tokens talk to each other

Input:
[
X \in \mathbb{R}^{B \times T \times d_{\text{model}}}
]

First, three learned linear maps:

[
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
]

Each still has shape:
[
[B, T, d_{\text{model}}]
]

Then reshaped into heads:
[
[B, h, T, d_{\text{head}}]
\quad\text{where}\quad
d_{\text{head}} = d_{\text{model}} / h
]

Now the *only* moment where shape meaningfully changes:

[
\text{Attention scores: } QK^\top
;\Rightarrow;
[B, h, T, T]
]

This is the “who attends to whom” matrix.

After softmax and weighting (V), everything collapses back to:

[
[B, T, d_{\text{model}}]
]

So attention temporarily creates a **token–token interaction matrix**, but it never leaves the block.

---

### 3.2 Feedforward network: no token mixing

The MLP is applied **independently to each token**:

[
\text{FFN}(x_t) = W_2 \sigma(W_1 x_t)
]

Shapes:
[
[B, T, d_{\text{model}}]
;\rightarrow;
[B, T, d_{\text{ff}}]
;\rightarrow;
[B, T, d_{\text{model}}]
]

No cross-token interaction here.
All mixing already happened in attention.

---

### 3.3 Residuals + normalization

Residual connections ensure:

[
\text{output} = X + \text{sublayer}(X)
]

This is why the shape *must* stay the same.
Transformers are iterative *representation refiners*.

---

## 4. Stacking layers: still the same shape

A transformer with (L) layers is just:

[
X^{(0)} \rightarrow X^{(1)} \rightarrow \dots \rightarrow X^{(L)}
]

Each (X^{(\ell)} \in \mathbb{R}^{B \times T \times d_{\text{model}}}).

So your abstraction:

```
output = transformer(tokens)
```

means:

> “Each token has been rewritten (L) times using global context.”

Nothing more. Nothing less.

---

## 5. What *is* the output, conceptually?

This depends entirely on **what you do next**.

The transformer itself outputs:
[
\text{hidden states} \in \mathbb{R}^{B \times T \times d_{\text{model}}}
]

What those *mean* depends on your head:

* Language modeling: each token predicts next token
* Classification: you pool or select a special token
* Diffusion: each token predicts noise / velocity
* Gene expression: each token predicts latent deltas or distribution parameters

The transformer does not decide this.
The **head** does.

---

## 6. Pooling is *not* part of the transformer

If you see something like:

* CLS token
* mean pooling
* attention pooling

Those are **post-transformer operations**.

Example:
[
[B, T, d] ;\rightarrow; [B, d]
]

That’s a design choice, not transformer logic.

This distinction matters a lot when you design biological models:
token-level outputs ≠ sample-level outputs.

---

## 7. Now the key question: what changes with LoRA / adapters?

Short answer: **nothing about the output shape changes.**

Long answer: the *function* changes, not the *type signature*.

---

### 7.1 LoRA in one sentence

LoRA replaces a weight matrix (W) with:

[
W_{\text{eff}} = W + \Delta W
\quad\text{where}\quad
\Delta W = A B
]

* (A \in \mathbb{R}^{d_{\text{out}} \times r})
* (B \in \mathbb{R}^{r \times d_{\text{in}}})
* (r \ll d_{\text{model}})

The key point:

[
W_{\text{eff}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
]

Same shape as before.

So when you apply LoRA to:

* (W_Q, W_K, W_V)
* output projection
* MLP layers

the **input/output tensor shapes are identical**.

---

### 7.2 Why this is conceptually important

LoRA is a *functional perturbation* of the model:

* It bends attention geometry
* It nudges feature subspaces
* It steers behavior

But it **does not change the interface**.

This is why you can:

* swap LoRA modules per task
* hot-load adapters
* compose multiple “skills”

without touching downstream code.

From a software-engineering perspective, this is gold.

---

## 8. Thinking in “type signatures” (a useful mental model)

Here’s a way to keep your sanity when models get big.

Think of a transformer as having a **type**:

[
\text{Transformer}[T, d] :
\text{Tokens}[T, d] \rightarrow \text{Tokens}[T, d]
]

LoRA, adapters, fine-tuning, freezing—none of these change the type.

Only **encoders, decoders, and heads** change types.

Examples:

* encoder: counts → tokens
* transformer: tokens → tokens
* decoder: tokens → distributions

This separation is exactly what makes foundation-model reuse possible.

---

## 9. Why this matters for your DiT + biology project

This abstraction unlocks a powerful design principle:

> Once gene expression is mapped into a token space, **all foundation-model machinery becomes legal.**

Diffusion, DiT, CFG, LoRA, adapters—they all operate on:
[
[B, T, d]
]

Your real design freedom lives in:

* how you *tokenize* biology
* how you *decode* outputs
* how you *condition* transformations

The transformer itself is just the universal mixer.

---

### A final anchoring sentence

A transformer does not generate meaning.
It **redistributes information across tokens while preserving shape**.

Everything interesting happens in how you *enter* and *exit* that space.

Next natural continuation: we can take one concrete biology example (say scRNA latent diffusion) and trace **exact tensor shapes end-to-end**, including conditioning paths and LoRA injection points, until nothing feels hand-wavy anymore.
