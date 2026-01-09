# PBMC Datasets for Generative Modeling

> **PBMC 3k and 68k are the MNIST of single-cell biology** — standardized, clean, and the right starting point for VAE, cVAE, and score-based models.

---

## What Are PBMC Datasets?

**PBMC** stands for **Peripheral Blood Mononuclear Cells** — immune cells circulating in blood: T cells, B cells, NK cells, monocytes, and dendritic cells.

Biologically, they're appealing because:

- They're diverse but well-studied
- They have strong, stereotyped transcriptional programs
- Cell types are separable yet overlapping (a perfect stress test for latent models)

Technologically, PBMC datasets come from **10x Genomics** and are widely used as benchmarks in single-cell analysis.

---

## PBMC 3k: The "Hello World" of scRNA-seq

**PBMC 3k** contains ~2,700 cells from a healthy donor, sequenced using droplet-based scRNA-seq.

### What You Get Per Cell

- A vector of **raw UMI counts** (~13,000–20,000 genes after filtering)
- Extreme sparsity (90–95% zeros)
- A total count per cell (library size)
- Latent biological structure visible even with simple models

### Why It's Perfect for This Roadmap

- **Small enough** to iterate fast
- **Large enough** to expose overdispersion
- **Cell types are known** and interpretable
- **Ideal for validating** NB vs ZINB likelihoods
- **Excellent for** latent space visualization (UMAP/t-SNE)

PBMC 3k lets you verify that your **ELBO, KL term, decoder parameterization, and library-size handling are correct** before scaling.

From a generative-modeling perspective, PBMC 3k is where you learn to *respect the data manifold*.

---

## PBMC 68k: Same Biology, Different Regime

**PBMC 68k** is the same *kind* of data, but in a different computational universe: ~68,000 cells.

What changes is not biology — it's **statistics and scaling**.

### Key Differences

- Many more rare cell states
- Much sharper estimates of dispersion parameters
- Clearer separation between biological and technical noise
- Enough data to expose posterior collapse issues in VAEs
- Enough scale to make diffusion / score models meaningful

### Where It Matters

- Amortized inference actually matters
- Batch effects start to bite
- Latent dimensionality becomes nontrivial
- Conditional models (cVAE) become obviously useful

If PBMC 3k asks *"does this work?"*, PBMC 68k asks *"does this still work under pressure?"*

---

## Why PBMC Is Ideal for Generative Expression Modeling

A core goal in computational biology — pursued by companies like Synthesize Bio, insitro, and others — is to:

> Generate **biologically realistic gene expression states** under different conditions

This enables studying treatment responses, predicting perturbation effects, and simulating counterfactual scenarios *in silico*.

PBMC datasets let you practice *exactly that* in miniature:

| Concept | PBMC Mapping |
|---------|--------------|
| **Condition** | Cell type (T cell, monocyte, etc.) |
| **Latent** | Cell state, activation, continuous variation |
| **Generation** | Simulate realistic immune profiles |
| **Counterfactual** | "What if this monocyte were a T cell?" |

That's not toy modeling — that's the same abstraction used in scVI, scGen, and industry-scale expression simulators.

### What PBMC Teaches You

- How much structure comes from the condition
- How much must live in the latent
- When the model cheats by encoding labels in z
- How overdispersion actually looks in practice

---

## How PBMC Fits Into Our Roadmap

Mapping this directly onto our **genai-lab roadmap**:

### VAE (Stage 1)

- **PBMC 3k**: Learn reconstruction + KL balance
- **PBMC 68k**: Test stability, latent capacity

### cVAE

- Condition on cell type
- Generate realistic expression given (z, cell_type)
- Perform label swapping to test disentanglement

### β-VAE

- Explore how much biological variation survives stronger bottlenecks

### Score Matching / Diffusion (Stage 3+)

- PBMC 68k gives enough density to meaningfully estimate scores
- Noise → denoise → recover biological structure

At each stage, PBMC acts as a *known reference system*. If the model fails here, it won't magically succeed on GTEx or TCGA.

---

## Critical Point: Keep Raw Counts

PBMC data **must stay in raw count space** if you're serious about generative modeling.

The correct approach (as in `data_preparation.md`):

- No log1p
- No CPM
- No normalization before modeling
- Library size treated explicitly

This is not pedantry. If you normalize first, you destroy the generative story and force the model to learn artifacts instead of biology.

PBMC is forgiving enough to show you this mistake clearly — another reason it's a great teacher.

---

## Big Picture Takeaway

PBMC 3k and 68k are not just datasets — they're **didactic instruments**.

They teach you:

- How generative assumptions meet biological reality
- How overdispersion emerges naturally
- How conditioning interacts with latent structure
- How scale changes model behavior

Once PBMC feels intuitive, moving to Tabula Sapiens, GTEx, or disease cohorts becomes an *engineering problem*, not a conceptual one.

### Next Steps

- Walk through PBMC preprocessing line-by-line
- Sketch the exact cVAE computational graph for PBMC
- Design evaluation metrics for realistic expression generation

---

## Is It Okay to Use Normalized Counts?

This is a *really* important question, and the confusion is extremely common — even among people who use NB/ZINB every day.

**Short answer**: Normalized counts look like counts, but they are no longer generated by a count process.

The long answer is where the insight lives.

---

### 1. What NB/ZINB Are Actually Modeling

A Negative Binomial model is not just a curve that fits integers. It encodes a **physical data-generation process**:

1. **True expression rate** of gene *g* in cell *i*
2. **Sequencing depth / capture efficiency** of that cell (library size)
3. **Sampling noise + biological variability**

Mathematically, the canonical scRNA-seq NB story is:

> "Given a cell with library size $\ell_i$, gene g produces counts $y_{ig}$ drawn from an NB distribution with mean proportional to $\ell_i$."

Written plainly:

- Counts increase when you sequence deeper
- Variance increases with the mean (overdispersion)
- Zero inflation comes from biology + dropout, not arithmetic tricks

That story is *true in the wet lab*. That's why NB works so well.

---

### 2. What Normalization Actually Does

Normalization (CPM, TPM, size-factor normalization, log1p, etc.) **rewrites the data** to answer a *different question*:

> "What would expression look like *if all cells had the same depth*?"

That is a **deterministic transformation** of the raw counts.

Example (CPM):

```text
y_ig  →  y_ig / ℓ_i × 10⁶
```

Key observation:

- $\ell_i$ (library size) is *removed*
- Depth variability is *collapsed*
- Values are now **ratios**, not samples from a counting process

You didn't just rescale the data. You **changed the random variable**.

---

### 3. Why "They're Still Counts" Is Misleading

After normalization:

- Values may still be non-negative
- They may even be integers (if you round)
- They may look "count-like"

But generatively, they are no longer counts.

| Question | Answer Type |
|----------|-------------|
| "How many molecules did I observe?" | **True count** |
| "What fraction of my sequencing budget went to this gene?" | **Normalized value** |

Those are *not the same random experiment*.

**NB/ZINB likelihoods assume:**

- Randomness comes from molecular sampling
- Variance depends on the mean *and* depth
- Library size is a latent or observed covariate

**After normalization:**

- Depth is fixed by construction
- Variance is artificially homogenized
- Mean–variance coupling is broken

So the likelihood is **wrong**, even if the curve "fits".

---

### 4. The Hidden Problem: Double-Using Library Size

This is the subtle killer.

If you:

1. Normalize counts to remove library size
2. Then train a model with an NB decoder

You've implicitly told the model:

> "Pretend depth doesn't matter — but also explain variance as if it did."

That contradiction forces the model to:

- Invent fake dispersion
- Misuse the latent space
- Leak technical effects into z
- Blur biology and noise

This is one of the main reasons people see:

- Posterior collapse
- Mushy latent spaces
- Poor counterfactuals

The model is trying to explain artifacts you injected.

---

### 5. Why Normalization *Is* Fine for Some Tasks

Normalization isn't evil. It's just task-specific.

**Appropriate for (descriptive tasks):**

- Clustering
- Visualization (UMAP / PCA)
- Differential expression heuristics
- Linear models assuming homoskedasticity

**Not appropriate for (generative tasks):**

- Asserting a data-generating process
- Likelihood-based evaluation
- Counterfactuals that must respect physics

That's why scVI, scGen, and industry-scale models **all operate on raw counts with explicit size factors**.

---

### 6. The Correct Way to "Normalize" in Generative Models

Instead of transforming the data, you transform the *model*.

The canonical trick:

> Keep raw counts; include library size as an offset or covariate

Conceptually:

- Decoder predicts a *rate*
- Library size scales that rate
- NB handles overdispersion naturally

This preserves:

- Correct variance structure
- Biological signal
- Valid sampling semantics

Nothing is thrown away.

---

### 7. A Mental Checksum

Ask this question:

> "Could I plausibly simulate raw sequencing data from this representation?"

- **Raw counts** → yes
- **Normalized counts** → no

If you can't simulate sequencing, you're not doing generative biology — you're doing regression on a convenience transform.

---

### 8. Why This Matters for Diffusion Models

This becomes *even more critical* for:

- Score matching
- Diffusion models
- Likelihood-based evaluation

Those methods assume:

- The data distribution is real
- Noise has a physical interpretation
- The model can walk backward to plausible samples

Normalized data breaks that chain completely.

---

## Bottom Line

**Normalization answers analytical questions. Raw counts answer generative questions.**

If your goal is:

- Counterfactual gene expression
- In silico experiments
- Realistic sampling
- Industry-grade generative models

Then raw counts + explicit library size is not a preference — it's a requirement.

This distinction is one of the big conceptual thresholds between *using* generative models and *understanding* them.
