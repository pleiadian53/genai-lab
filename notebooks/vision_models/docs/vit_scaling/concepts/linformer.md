# Linformer

## One-sentence version

Linformer makes self-attention cheaper by projecting the key and value sequences into a lower-dimensional sequence length, replacing full `N × N` attention with something closer to `N × k`, where `k << N`.

## Why it exists

Standard attention computes:

```text
softmax(QKᵀ)V
```

If `Q` and `K` each have sequence length `N`, then `QKᵀ` is:

```text
N × N
```

That is the quadratic bottleneck.

Linformer starts from the hypothesis that attention is often approximately low-rank. In plain language: the full attention matrix may contain redundancy.

## Core mechanism

Instead of using full keys and values:

```text
K: N × d
V: N × d
```

Linformer learns projections along the sequence dimension:

```text
K_compressed: k × d
V_compressed: k × d
```

where:

```text
k << N
```

Then attention compares each query against a compressed set of keys:

```text
QK_compressedᵀ
```

The matrix is now:

```text
N × k
```

instead of:

```text
N × N
```

## Intuition

Imagine reading a long document. Instead of comparing every sentence to every other sentence, you first compress the document into a smaller set of representative summaries. Each sentence then attends to those summaries.

That is the Linformer flavor:

```text
full token memory → compressed token memory
```

## What it buys you

Linformer can reduce attention complexity from quadratic toward linear in sequence length, assuming `k` is treated as fixed or much smaller than `N`.

This is useful when long sequences make full attention too expensive.

## What it costs

The compression is a bottleneck.

If the task truly needs fine-grained pairwise token interactions, projecting keys and values down too aggressively can lose information.

So the tradeoff is:

```text
speed/memory efficiency ↔ fidelity of attention structure
```

## How it connects to ViT scaling

For high-resolution ViT, the sequence length is the number of image patches.

Vanilla ViT asks:

```text
Does every patch need to compare with every other patch directly?
```

Linformer answers:

```text
Maybe each patch only needs to compare against a compressed representation of all patches.
```

## Ripple and connect

Related ideas:

- **Low-rank approximation**: represent a large matrix with fewer latent dimensions.
- **Dimensionality reduction**: preserve the important structure while shrinking the representation.
- **Nyström methods**: approximate large matrices from selected landmarks.
- **Bottleneck layer**: force information through a smaller channel to gain efficiency.
- **Attention compression**: reduce the memory over which queries attend.

Useful idioms/phrases:

- “Linformer compresses the memory side of attention.”
- “It replaces all-token comparison with comparison against a learned summary.”
- “The bet is that attention is redundant enough to be low-rank.”

## Interview-ready explanation

Linformer is an efficient transformer variant that reduces the quadratic cost of self-attention by projecting keys and values along the sequence dimension to a smaller length `k`. Instead of forming an `N × N` attention matrix, it forms roughly an `N × k` matrix. The method relies on the assumption that attention is approximately low-rank, so the full token interaction structure can be compressed without losing too much information.

## Research notes from the Linformer paper

Linformer starts from the observation that standard self-attention uses `O(n²)` time and space with respect to sequence length. The paper's central claim is that the self-attention matrix can be approximated by a **low-rank matrix**, allowing attention complexity to be reduced toward `O(n)` in time and space.

The mechanism is learned projection of keys and values along the sequence dimension. Queries still exist at the original length, but they attend to a compressed representation of keys/values.

### What is actually being compressed?

Not the hidden dimension first. The important compression is along the **sequence length axis**:

```text
sequence length n → projected length k
```

That is why the attention score shape changes from approximately:

```text
n × n
```

to:

```text
n × k
```

### What to remember

Linformer is a low-rank bet: many attention maps may not need full rank to preserve useful token interactions. The risk is that some tasks may need the fine structure thrown away by the projection.

## Source

- Wang et al., “Linformer: Self-Attention with Linear Complexity” — https://arxiv.org/abs/2006.04768
