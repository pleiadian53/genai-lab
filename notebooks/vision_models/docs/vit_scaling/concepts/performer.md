# Performer

## One-sentence version

Performer makes attention scale linearly by approximating softmax attention with random feature maps, avoiding explicit construction of the full `N × N` attention matrix.

## Why it exists

Standard attention is:

```text
softmax(QKᵀ)V
```

The expensive part is forming `QKᵀ`, which compares every query token to every key token.

Performer asks whether we can compute something close to softmax attention without materializing that pairwise matrix.

## Core mechanism

Performer uses a kernel view of attention.

Very roughly, it maps queries and keys through a feature transformation:

```text
Q → φ(Q)
K → φ(K)
```

Then attention can be reorganized using associativity:

```text
φ(Q) (φ(K)ᵀ V)
```

instead of:

```text
(φ(Q) φ(K)ᵀ) V
```

The key is that `φ(K)ᵀ V` can be computed before multiplying by every query, avoiding the full `N × N` matrix.

Performer’s specific method is commonly associated with FAVOR+, which uses positive random features to approximate softmax attention.

## Intuition

Think of vanilla attention as building a huge table of pairwise relationships:

```text
token i ↔ token j
```

Performer avoids building the table. It maps tokens into a feature space where the same kind of weighted aggregation can be computed in a more factorized way.

It is like replacing an all-pairs conversation with a clever shared coordinate system.

## What it buys you

The goal is linear complexity in sequence length:

```text
O(N)
```

rather than:

```text
O(N²)
```

This is attractive for long sequences, including high-token vision inputs.

## What it costs

Performer is approximate.

The quality depends on the random feature approximation and implementation details. In practice, the tradeoff is:

```text
much cheaper attention ↔ approximation error / numerical considerations
```

## How it connects to ViT scaling

In high-resolution ViT, the problem is that every patch compares with every other patch.

Performer says:

```text
Keep global attention in spirit, but compute it through an approximation that avoids the full matrix.
```

This differs from Swin. Swin changes the attention pattern to local windows. Performer tries to preserve global attention more directly, but with an efficient approximation.

## Ripple and connect

Related ideas:

- **Kernel trick**: compute relationships through feature maps instead of explicit pairwise comparison.
- **Random features**: approximate kernels using randomized projections.
- **Associativity**: change the multiplication order to avoid a huge intermediate matrix.
- **Linear attention**: family of methods that avoid quadratic attention.
- **Approximation vs sparsity**: Performer approximates dense global attention; Swin sparsifies/restricts attention locally.

Useful idioms/phrases:

- “Performer factorizes attention through random features.”
- “It keeps global mixing but avoids the all-pairs matrix.”
- “The trick is not just fewer tokens; it is changing the algebra.”

## Interview-ready explanation

Performer is an efficient transformer architecture that approximates softmax attention using random feature maps, often described through FAVOR+. By rewriting attention in a kernelized form, it avoids explicitly constructing the `N × N` attention matrix and computes attention in roughly linear time. Compared with local-attention methods like Swin, Performer aims to preserve global attention behavior through approximation rather than restricting attention to windows.

## Research notes from the Performer paper

The Performer paper is careful about what it is *not* assuming. It aims to approximate full softmax attention with linear time and space **without relying on sparsity or low-rankness**.

Its core method is FAVOR+: **Fast Attention Via positive Orthogonal Random features**. FAVOR+ approximates softmax attention kernels so the computation can be reorganized without explicitly forming the full attention matrix.

### Why this is different from Linformer

Linformer says:

```text
The attention matrix is approximately low-rank, so compress keys/values.
```

Performer says:

```text
Softmax attention is kernelizable, so approximate the kernel and reorder the computation.
```

This is why Performer is usually described as **kernelized linear attention**, not low-rank attention.

### What to remember

Performer tries to keep the spirit of dense global attention while avoiding the materialized `N × N` matrix. The cost is that attention is now an estimator/approximation, so implementation details, random feature quality, and numerical stability matter.

## Source

- Choromanski et al., “Rethinking Attention with Performers” — https://arxiv.org/abs/2009.14794
