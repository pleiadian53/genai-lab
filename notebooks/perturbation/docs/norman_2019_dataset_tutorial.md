# Norman et al. 2019 Perturb-seq — Dataset & Biology Primer

A tutorial for the flagship dataset we'll use across milestones P1-P5.
The goal is that by the end of this doc you have:

1. A mental model of what the data actually *is* (biology + experimental
   design)
2. An understanding of *why* this dataset is the canonical benchmark for
   perturbation response prediction
3. A justified rationale for every QC threshold the loader applies
4. A feel for how published methods evaluate on it, so our P2-P5 benchmarks
   land on common ground with the literature

This is a **background** document — no code. Runnable exploration lives in
[`notebooks/perturbation/01_norman_eda.ipynb`](../01_norman_eda.ipynb)
(populated when P1 smoke-tests successfully). Loader code lives in
[`src/genailab/applications/perturbation/data/norman.py`](../../../src/genailab/applications/perturbation/data/norman.py).

---

## 0. Background — Biology Primer

Skip this section if "K562 CRISPRa Perturb-seq" is already self-evident.

### 0.1 What is K562?

K562 is an **immortalized human cell line** derived in 1975 from a
patient with chronic myeloid leukemia (CML) in blast crisis. Three things
make it the single most widely used cell line in functional genomics:

- **Fast, robust growth** — doubles in ~24 h, tolerant to transfection
- **Known driver lesion** — carries the `BCR-ABL` fusion (the hallmark of
  CML); proliferation is largely addictive to BCR-ABL kinase signaling
- **Erythroleukemia phenotype** — displays partial erythroid /
  megakaryocytic differentiation potential, expresses a broad gene
  repertoire that captures many cellular pathways

K562 is the reference cell line for the ENCODE consortium, most CRISPR
screen pipelines, and the Perturb-seq papers we'll benchmark against.

**What K562 is *not***: a primary cell. It's a cancer cell line that has
drifted through decades of passage. Conclusions about normal hematopoiesis
or drug-target discovery drawn from K562 need validation in primary cells
or organoids. For method development this is fine; for biological claims
it's a caveat.

### 0.2 What does "Perturb-seq" mean?

**Perturb-seq = CRISPR perturbation + single-cell RNA sequencing**, in a
single experiment. Each cell receives a CRISPR guide RNA (sgRNA) targeting
some gene, and the 10x Genomics readout captures *both*:

1. The cell's **transcriptome** (standard scRNA-seq, ~20,000 genes)
2. The **identity of the guide RNA** the cell received (via a separate
   barcode sequence, captured in the same 10x bead)

So for every one of ~100,000 cells you know "this cell had its gene X
activated" AND "here are its expression levels across all genes." That
coupling is the whole point — it's how you read out the *effect* of a
perturbation on transcription at single-cell resolution, at scale.

Originating paper: Dixit, Parnas, Li et al. 2016 (*Cell*).

### 0.3 CRISPR activation (CRISPRa), not knockout

Norman et al. 2019 uses **CRISPRa** — a variant of CRISPR that
**up-regulates** the target gene rather than knocking it out. The
mechanism:

- Catalytically dead Cas9 (dCas9 — can't cut DNA)
- Fused to a transcriptional activator (VPR: VP64-p65-Rta, or similar)
- Guided to the target gene's promoter by the sgRNA
- → Recruits the transcription machinery, **increases expression** of the
  target gene

Compare with the two other common variants:

| Variant | Effect | Typical use |
|---------|--------|-------------|
| CRISPR knockout (KO) | Double-strand break → frameshift → loss of function | "What happens without gene X?" |
| CRISPRi (interference) | dCas9-KRAB → represses transcription | Milder loss of function |
| **CRISPRa (what Norman uses)** | dCas9-VPR → activates transcription | **"What happens with *extra* gene X?"** |

For Norman 2019, CRISPRa was the right tool because the scientific
question was about **overexpression combinations** — pairs of genes
simultaneously activated to study synergy/buffering.

### 0.4 Why combinatorial perturbations?

Biology is combinatorial — gene A and gene B together don't necessarily
do A+B. They can:

- **Buffer** — A+B effect < A + B (one compensates for the other)
- **Epistasis** — A+B effect = A alone or B alone (one is dominant)
- **Synergize** — A+B effect > A + B (true emergent phenotype)
- **Act independently** — A+B effect ≈ A + B (additive)

Mapping which gene pairs fall into which category is the core of
**genetic interaction** characterization. Classical genetics (Saccharomyces
yeast, e.g., Costanzo et al. 2010) built large genetic-interaction maps by
brute-force double knockouts. Norman et al. 2019 brought this approach to
mammalian cells at scale with single-cell resolution.

---

## 1. The Norman et al. 2019 Experiment

**Citation**: Norman TM, Horlbeck MA, Replogle JM, Ge AY, Xu A, Jost M,
Gilbert LA, Weissman JS. *Exploring genetic interaction manifolds
constructed from rich single-cell phenotypes.* Science 365: 786-793 (2019).

### Design

- **Cells**: K562
- **Perturbation tool**: CRISPRa (dCas9-SunTag + scFv-VP64)
- **Targeted genes**: ~200 known regulators of cell state, differentiation,
  and identity, chosen for their potential to drive morphological /
  transcriptional changes
- **Design**:
  - **Singleton perturbations** (one guide per cell): ~100-150 distinct
    target genes
  - **Pair perturbations** (two guides per cell): ~300-600 pairs (sampled
    from the singleton set)
  - **Controls**: non-targeting guides (`NT`) used as baseline
- **Readout**: 10x Genomics 3' scRNA-seq with a perturbation-capture
  amplicon library to identify which guide(s) each cell received

### Resulting dataset size

- **~111,000 cells** after QC (per the paper)
- **~20,000 genes** before HVG selection
- Each cell carries a perturbation label of the form `GeneA` (singleton),
  `GeneA+GeneB` (pair), or `ctrl` (non-targeting control)

Exact numbers vary with preprocessing choices; scPerturb's standardized
re-release may differ slightly from the paper's reported counts. The
loader's QC step produces reproducible numbers regardless.

### Why this specific dataset?

Several things make Norman 2019 the reference Perturb-seq benchmark:

1. **Balanced design** — both singletons and pairs, at depth
2. **Well-characterized targets** — cell-identity regulators with known
   biology, so sanity checks ("does perturbing GATA1 push cells toward
   erythroid state?") are possible
3. **Depth per perturbation** — hundreds of cells per singleton, tens to
   hundreds per pair — enough to estimate effects reliably
4. **Publicly re-released via scPerturb** — standardized preprocessing,
   trivial to load

---

## 2. What the Dataset Contains

After loading via scPerturb, you get an `AnnData` object with roughly
this structure:

```
AnnData
├── X: csr_matrix, shape (~111000, ~20000), dtype int32   # raw UMI counts
├── obs: DataFrame indexed by cell barcode
│   ├── perturbation        — string label, e.g. "GATA1", "GATA1+FOXA1", "ctrl"
│   ├── perturbation_2      — (sometimes) split columns for multi-target analysis
│   ├── n_guides            — number of distinct guide RNAs per cell (1 or 2)
│   ├── gemgroup / batch    — technical batch info
│   └── (possibly) cell_type, tissue_type, condition
├── var: DataFrame indexed by gene symbol
│   └── ensembl_id, feature_type, ...
└── layers:                 — typically empty in the raw release
```

Key structural points:

- **Counts are raw integers.** Never normalized in the distributed file.
  The whole value of this dataset for generative modeling depends on
  preserving this.
- **Perturbation label is a string**, not a pair of columns. Pair
  perturbations are encoded as `"A+B"`. Our loader splits these into
  primary/secondary columns for downstream use.
- **Controls are labeled** with a fixed string (most commonly `"ctrl"`
  or `"control"`, sometimes `"NT"` for non-targeting). These are
  ~5-10% of cells and serve as the reference state for all
  perturbation-effect calculations.
- **Guide-level information** — which specific sgRNA (there are often
  2-3 guides per gene) was used — may or may not be in the release.
  Guide-level detail matters if you want to aggregate across guides for a
  target gene, or flag guides that didn't work.

The loader's QC step inspects the actual column names and reports what's
available.

---

## 3. Why This Dataset Is a Computational Biology Workhorse

Roughly five overlapping research agendas use this dataset as a
benchmark. Understanding which one you're pursuing frames the evaluation
choices.

### 3.1 Perturbation response prediction

**The core task.** Given:
- Baseline (control) expression distribution
- A new perturbation specification (target gene, or gene pair)

Predict the **perturbed expression distribution**. Measured as the
per-gene Pearson correlation between predicted and observed *mean*
expression shifts on held-out perturbations. This is our flagship's
primary metric (target ~0.80+ per scGen/CPA).

### 3.2 Compositional generalization

**The harder task.** Train on **singletons only**, then predict
**pair** effects. Tests whether the model learned a compositional
structure (gene A's direction of effect + gene B's direction of effect
→ combined direction) or merely memorized observed responses.

Success here is a strong signal that the model has captured something
biological rather than fitting empirical correlations. CPA and scPPDM
explicitly report compositional generalization scores.

### 3.3 Genetic interaction characterization

**The biology question.** For every pair (A, B):

- Does A+B buffer, synergize, or behave additively?
- Is the interaction asymmetric?
- Can you cluster pairs by their interaction signature?

Less "predict the response" and more "characterize the manifold."
Norman 2019's own analysis sits here. Not our primary focus but
downstream users (and reviewers) care.

### 3.4 Latent factor learning

**What are the axes of variation?** If you project all cells (controls +
every perturbation) into a low-dimensional latent, how many meaningful
directions emerge, and do they correspond to known biological programs
(proliferation, differentiation, apoptosis, erythroid identity, ...)?

Related to interpretability. CPA's disentangled latents are designed
around this framing.

### 3.5 Drug discovery via phenocopying

**Applied use.** If perturbing gene X produces an expression signature
similar to the effect of drug D (from LINCS/CMap), then X is a candidate
target or pathway involved in D's mechanism. This is why pharma cares
about Perturb-seq. Not a direct benchmark metric but a downstream
application that motivates the accuracy requirements.

---

## 4. Methods That Benchmark on This Dataset

The perturbation-prediction literature uses Norman 2019 as the de facto
benchmark. The main methods, ordered roughly by publication:

| Year | Method | Approach | Held-out Pearson* |
|------|--------|----------|-------------------|
| 2019 | **scGen** | Conditional VAE; perturbation as latent offset | ~0.80 |
| 2023 | **CPA** (Compositional Perturbation Autoencoder) | Disentangled latents: perturbation, dose, covariate | ~0.85 |
| 2023 | **GEARS** | Graph-based (protein-protein interaction prior) | ~0.82 |
| 2024 | **scFoundation + perturbation head** | Foundation-model embedding + supervised head | ~0.83 |
| 2025 | **scPPDM** | Diffusion model for Perturb-seq | ~0.87 |

\* Exact numbers vary by protocol (which splits, which perturbation
subset). Treat as ballpark; the comparison is relative.

**Our stack** (flagship roadmap):
- P2: **CVAE_NB with perturbation conditioning** — match scGen (~0.80)
- P3: **JEPA** (self-supervised latent-space prediction) — target ~0.85 (CPA)
- P4: **Latent diffusion on JEPA latent** — target ~0.87 (scPPDM), plus
  uncertainty quantification (not reported by scPPDM)

---

## 5. QC Decisions — What and Why

The `qc_norman()` function in the loader applies these thresholds. Each
one has a biological/technical rationale.

### 5.1 Cell-level filters

| Threshold | Default | Why |
|-----------|--------:|-----|
| `min_genes` | 200 | Cells expressing fewer than 200 genes are likely empty or damaged droplets. Standard across all 10x Genomics scRNA-seq pipelines. Sensitivity to this threshold is low — 200 to 500 doesn't change downstream results much. |
| `pct_mt_max` | 20 | Cells with >20% mitochondrial counts are dying (cytoplasmic RNA leaks before mitochondrial RNA). For **K562 specifically**, 20% is higher than the 10-15% used for primary PBMCs because cell lines have different MT dynamics — K562 shows higher baseline MT fraction even in healthy cells. The Norman paper used ~20%. |
| `min_guides` | 1 | Cells with no detectable guide RNA can't be assigned a perturbation. Sometimes pre-filtered in the scPerturb release. |

### 5.2 Perturbation-level filters

| Threshold | Default | Why |
|-----------|--------:|-----|
| `min_cells_per_perturbation` | 30 | Perturbations with <30 cells give unreliable effect estimates and confound train/val/holdout splits. 30 is on the low end; some methods require 50 or 100. We use 30 to retain more perturbations for compositional experiments. |

### 5.3 Gene-level filters

| Threshold | Default | Why |
|-----------|--------:|-----|
| `min_cells` | 3 | Genes detected in <3 cells carry no signal and inflate dimensionality without information. Standard scRNA-seq practice. |

### 5.4 Multiplet filter (optional)

Some cells receive two or more guide RNAs unintentionally during the
transduction step. Norman 2019 is explicitly designed for pair
perturbations, so "two guides" is often *intended* (the pair condition)
rather than a multiplet. The loader distinguishes:

- 1 guide → singleton perturbation — valid
- 2 guides at expected pair barcodes → pair perturbation — valid
- 3+ guides OR 2 guides at unexpected positions → multiplet, likely drop

scPerturb's release typically handles this; the loader preserves whatever
annotation is present and reports the distribution.

### 5.5 What we deliberately *don't* do

- **Don't apply HVG selection at load time.** HVG choice is
  model-dependent (2000 HVGs for a baseline, full gene set for a
  foundation model). Loader returns filtered-but-full gene space; the
  model-side code selects HVGs.
- **Don't normalize.** Ever. The whole point of NB/ZINB decoders is
  that they model raw counts. Normalization is reserved for descriptive
  analyses (UMAP, DE heuristics in the EDA notebook) and always on a
  copy.
- **Don't compute PCA/UMAP in the loader.** These are model/analysis
  choices, not data-cleaning steps.

---

## 6. Train / Val / Holdout Splitting Conventions

How you split determines what generalization you measure. Three
standard strategies:

### 6.1 `split="cell"` — random cell-level (stratified by perturbation)

Holds out a fraction of cells within each perturbation. Measures: can
the model reconstruct responses it has seen during training. Useful as a
sanity check but **leaky** — the perturbation itself is in the training
set, so this doesn't test generalization.

P1 uses this split for the initial smoke test.

### 6.2 `split="perturbation"` — held-out perturbations

Entire perturbations (singletons or pairs) are held out — none of their
cells appear in training. Measures: **can the model predict the effect
of an unseen perturbation**. This is scGen/CPA's primary evaluation
protocol.

This is the metric we target for P2's ~0.80 Pearson.

### 6.3 `split="combination"` — compositional generalization

Train on all singletons + a subset of pairs. Hold out the *remaining
pairs* and evaluate whether the model can compose pair effects from the
singletons it saw. Measures: **does the model know how to combine**.

This is the scPPDM / CPA signature evaluation. It's the hardest split
and the most discriminating between memorization and understanding.

### 6.4 What P1 delivers

P1's loader implements `split="cell"` (trivial random split). The other
two strategies depend on inspecting the actual perturbation label
structure — we'll add them after P1's first run tells us what the labels
look like in the scPerturb release.

---

## 7. Expected Shapes at Each Stage

Use these as landmarks when you run P1. If your numbers diverge by more
than ~5% from these, check whether the scPerturb release has been
updated or whether a QC threshold needs adjustment.

| Stage | Cells | Genes | Notes |
|-------|------:|------:|-------|
| Raw download | ~111,000 | ~22,000 | Pre-QC counts as distributed by scPerturb |
| After cell-level QC | ~100,000 | ~22,000 | Drops damaged + empty cells |
| After gene-level QC | ~100,000 | ~14,000 | Drops never-detected genes |
| After perturbation-level QC | ~95,000 | ~14,000 | Drops low-cell-count perturbations (<30 cells) |
| After HVG (model-side only) | ~95,000 | 2,000 | Done inside modeling scripts, not the loader |

These numbers are approximate. The first real P1 run establishes the
authoritative values.

---

## 8. Accessing the Data

### Source: scPerturb.org (default)

[scPerturb](https://scperturb.org/) is a community-maintained aggregator
that re-releases Perturb-seq datasets with consistent preprocessing
conventions (standard column names, raw counts preserved, unified
AnnData schema). Norman 2019 is one of their flagship releases.

The loader fetches the `.h5ad` file (~1-2 GB) from scPerturb's CDN and
caches it under
`data/scrna/perturb_seq/norman_2019/` per the genai-lab data convention.

### Alternative sources (not used)

- **GEO (GSE133344)** — raw 10x matrices from the paper authors. Requires
  assembling from tar archives. More work, same data.
- **CellxGene** — has the dataset in their browser but loading it
  programmatically requires their Python client. Not worth the extra
  dependency.
- **figshare / Zenodo** — various intermediate releases. scPerturb is
  the cleanest maintained version.

### On-disk size

- Raw `.h5ad`: ~1-2 GB
- In-memory (`AnnData` with sparse X): ~500 MB for 111k × 22k @ ~5% density
- Dense representation would be ~10 GB — we never densify the full matrix.

### Staging to pod

For pod runs, stage the file to the RunPod network volume once:

```bash
python ops/provision_cluster.py --stage-data \
    --data-path scrna/perturb_seq/norman_2019
```

Subsequent pod provisions mount the volume instantly.

---

## 9. Related Datasets — When to Scale Up

Norman 2019 is the starter dataset. Other Perturb-seq resources become
relevant as the flagship matures:

| Dataset | Scale | When to use |
|---------|------:|-------------|
| **Norman 2019** | 111k cells, ~150 perturbations + pairs | Baseline development (this project's P1-P5) |
| **Replogle 2022** | ~2.5M cells, essential-gene screen | Scaling up, broader biology |
| **Replogle 2020** (GWPS) | ~2.5M cells, genome-wide | Foundation-model pretraining |
| **Adamson 2016** | ~15k cells, UPR pathway | Small, focused benchmark |
| **Dixit 2016** | 200k cells (multiple experiments) | Origins of Perturb-seq; historical comparison |

For this flagship we stay on Norman 2019 through P5. Scale-up to
Replogle is a future direction (not on the 6-week roadmap).

---

## 10. References

**Primary paper**
- Norman TM et al. (2019). Exploring genetic interaction manifolds
  constructed from rich single-cell phenotypes. *Science* 365: 786-793.
  [doi:10.1126/science.aax4438](https://doi.org/10.1126/science.aax4438)

**Perturb-seq methodology (originating paper)**
- Dixit A, Parnas O, Li B et al. (2016). Perturb-seq: Dissecting
  Molecular Circuits with Scalable Single-Cell RNA Profiling of Pooled
  Genetic Screens. *Cell* 167: 1853-1866.

**Benchmarks on this dataset**
- Lotfollahi M et al. (2019). scGen predicts single-cell perturbation
  responses. *Nature Methods* 16: 715-721.
- Lotfollahi M et al. (2023). Predicting cellular responses to complex
  perturbations in high-throughput screens. *Molecular Systems Biology*
  19: e11517. (CPA)
- Roohani Y, Huang K, Leskovec J. (2024). GEARS: predicting
  transcriptional outcomes of novel multi-gene perturbations. *Nature
  Biotechnology*.
- scPPDM (2025, arXiv preprint) — diffusion for Perturb-seq.

**Data aggregator**
- Peidli S et al. (2024). scPerturb: harmonized single-cell perturbation
  data. *Nature Methods* 21: 531-540.
  [scperturb.org](https://scperturb.org/)

**Background: K562 and CRISPRa**
- Gilbert LA et al. (2014). Genome-scale CRISPR-mediated control of gene
  repression and activation. *Cell* 159: 647-661.
- Andersson LC, Nilsson K, Gahmberg CG. (1979). K562 — a human
  erythroleukemic cell line. *International Journal of Cancer* 23:
  143-147. (Original K562 derivation)

---

## Related in genai-lab

- [`examples/perturbation/README.md`](../../../examples/perturbation/README.md) — P1-P5 milestone roadmap
- [`docs/applications/perturbation_prediction.md`](../../../docs/applications/perturbation_prediction.md) — application-level methodology doc
- [`notebooks/vae/docs/pbmc_dataset_tutorial.md`](../../vae/docs/pbmc_dataset_tutorial.md) — sibling dataset tutorial (PBMC, simpler)
- [`src/genailab/applications/perturbation/data/norman.py`](../../../src/genailab/applications/perturbation/data/norman.py) — the loader this tutorial motivates
