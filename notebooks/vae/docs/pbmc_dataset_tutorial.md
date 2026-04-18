# PBMC Datasets — A Practical Tutorial

**Companion to** [`notebooks/vae/02_pbmc3k_cvae_nb.ipynb`](../02_pbmc3k_cvae_nb.ipynb)

This document walks through the PBMC 3k dataset the notebook uses, explains what you should expect to see at each preprocessing step, and maps out the larger PBMC datasets you'll want to graduate to once the CVAE_NB baseline works.

> **Theory companion**: [`docs/datasets/gene_expression/PBMC.md`](../../../docs/datasets/gene_expression/PBMC.md) covers the *why* (why raw counts, why NB/ZINB, why PBMC is pedagogically ideal). This tutorial covers the *how* — loading, inspecting, QC — and the *what next* — scaling to larger PBMC datasets.

---

## 0. Background: What Are PBMCs, and What Are We Measuring?

### 0.1 What is a PBMC?

*PBMC* stands for **Peripheral Blood Mononuclear Cell** — the subset of blood cells that have a single round nucleus. When a blood sample is centrifuged over a density gradient (Ficoll separation), PBMCs form a distinct layer that can be harvested cleanly.

The PBMC fraction contains the core adaptive and innate immune cells:

| Cell type | Role | Typical fraction |
|-----------|------|-----------------|
| CD4+ T cells (helper) | Orchestrate adaptive immune response | ~25–35% |
| CD8+ T cells (cytotoxic) | Kill infected / tumour cells | ~15–20% |
| B cells | Produce antibodies | ~5–15% |
| NK cells (Natural Killer) | Innate cytotoxic response | ~5–15% |
| CD14+ Monocytes | Phagocytosis, antigen presentation | ~10–20% |
| FCGR3A+ Monocytes | Patrolling monocytes (less common) | ~5% |
| Dendritic cells | Antigen presentation, bridge innate/adaptive | ~1–2% |
| Megakaryocytes | Platelet precursors (rare contaminant) | <1% |

**What's excluded**: red blood cells (no nucleus), platelets (cell fragments), and granulocytes (neutrophils, eosinophils — multi-lobed nucleus, lost in the density step).

### 0.2 Why PBMCs?

PBMCs are the canonical benchmark dataset for single-cell genomics for three practical reasons:

1. **Accessibility.** A standard blood draw yields millions of PBMCs. No surgery, no biopsy, no animal sacrifice.
2. **Well-characterised biology.** Every major immune cell type has known *marker genes* (e.g., `CD3D` for T cells, `MS4A1` for B cells, `LYZ` for monocytes). This lets you validate whether a model learned biologically coherent representations — if CD4 T cells and CD8 T cells cluster separately, your latent space is capturing real biology.
3. **Clinical relevance.** Immune dysregulation underlies HIV, autoimmunity, cancer, COVID-19, and most inflammatory diseases. Models trained on PBMCs directly translate to disease research.

### 0.3 What Does Single-Cell RNA Sequencing (scRNA-seq) Measure?

Every cell in the body contains the same DNA. What differs between a T cell and a monocyte is **which genes are actively transcribed** into messenger RNA (mRNA) at any given moment. This is called *gene expression*.

Single-cell RNA sequencing captures a snapshot of that expression state:

```
Cell → lyse → capture mRNA → reverse transcribe to cDNA → sequence → count
```

The result is a **cell × gene count matrix**: one row per cell, one column per gene, each entry recording how many mRNA molecules of that gene were detected in that cell.

### 0.4 What Does Each Number in the Matrix Mean?

The values in `.X` are **UMI counts** (Unique Molecular Identifier counts) — not raw sequencing reads.

A UMI is a short random barcode (6–12 nucleotides) that is ligated to each captured mRNA molecule *before* PCR amplification. During sequencing, the PCR produces thousands of copies of each molecule, but all copies from the same original molecule share the same UMI. The pipeline deduplicates by UMI, so:

> **1 UMI = 1 original mRNA molecule detected in that cell.**

Concretely:
- A cell with `MALAT1 = 47` had 47 MALAT1 mRNA molecules detected — a highly expressed housekeeping gene.
- A cell with `CD3D = 3` had 3 CD3D mRNA molecules detected — low, suggesting this might not be a T cell.
- A cell with `CD3D = 0` had no CD3D mRNA detected. This could mean the gene is truly silent *or* that the molecule was present but not captured (technical dropout).

**UMI counts are not read counts.** A read count of 0 is always 0 UMI, but a read count of 5000 might correspond to 3 UMIs if those 3 molecules were amplified heavily. UMI counts are the deduplication that makes comparisons between cells meaningful.

### 0.5 Why Is the Matrix So Sparse (94% Zeros)?

The 2700 × 32738 raw matrix has roughly **1.7 million non-zero entries out of ~88 million total** — about 94% zeros. This is not a data quality problem; it is a structural property of single-cell data:

- **Biological zeros**: Most genes are truly silent in any given cell at any given moment. A monocyte simply does not express T-cell receptor genes.
- **Technical dropout**: The capture efficiency of the 10x droplet-based protocol is ~10–20%. Many mRNA molecules are lost during lysis, reverse transcription, or sequencing. A gene with 1–2 copies in a cell has a high probability of generating a zero even when present.

This mixture of true zeros and dropout zeros is precisely why the **Zero-Inflated Negative Binomial (ZINB)** decoder is more appropriate than a Gaussian: it explicitly models both sources of zero.

### 0.6 What Does the Shape 2700 × 32738 Mean?

| Dimension | What it represents |
|-----------|-------------------|
| **2700 rows (cells)** | Each row is one cell barcode — a unique droplet captured during the 10x run. A "cell barcode" is the short DNA sequence that identifies which droplet a given mRNA molecule came from. |
| **32738 columns (genes)** | Each column is one human gene (from the GRCh38 reference genome used for alignment). This is essentially all annotated protein-coding and non-coding RNA genes in the human genome. |

After QC and HVG selection you reduce to ~2638 cells × 2000 genes. The column reduction (32k → 2k) removes genes with near-zero variance across cells — they carry no discriminative signal about cell identity.

### 0.7 Why NB and Not Gaussian?

A Gaussian distribution assumes continuous, unbounded, symmetric values. UMI counts violate all three:
- **Discrete and non-negative** — you cannot have 2.7 mRNA molecules.
- **Highly skewed** — most genes have a count of 0 or 1; a few genes (MALAT1, ribosomal genes) have counts in the hundreds.
- **Overdispersed** — the variance is much larger than the mean, unlike a Poisson distribution where variance = mean.

The **Negative Binomial (NB)** distribution handles all of this: it models non-negative integers, accommodates heavy right tails, and has a free dispersion parameter $\theta$ that captures the extra variance beyond Poisson. The CVAE_NB decoder outputs NB parameters $(r, \theta)$ per gene per cell, making it a generatively correct model for this data.

---

## 1. What You Get From `sc.datasets.pbmc3k()`

The one-liner in the notebook hides a lot. Here's what it actually downloads and returns:

```python
import scanpy as sc
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()
```

### Provenance
- **Source**: 10x Genomics public demo dataset, released 2016
- **Donor**: Single healthy donor
- **Chemistry**: 10x Chromium **v1** (3' scRNA-seq) — older chemistry, noisier than v3
- **Tissue**: Frozen PBMCs

### Shape you should see
```
Raw data: 2700 cells, 32738 genes
```

If the numbers are different, something changed upstream (scanpy version, cache corruption). This exact shape is the canonical one you see in every scanpy tutorial.

### AnnData structure at this point
```
AnnData object
├── X: csr_matrix, shape (2700, 32738), dtype int32  ← raw UMI counts
├── obs: DataFrame (barcodes as index) — mostly empty initially
├── var: DataFrame (gene symbols as index) with 'gene_ids' (Ensembl IDs)
├── uns: empty
├── obsm/varm/layers: empty
```

Key properties:
- `.X` holds **integer UMI counts**, sparse (~94% zeros)
- Genes are indexed by **symbol** (e.g., `MALAT1`, `CD3D`) — that's why `var_names_make_unique()` matters (a handful of symbols collide)
- No cell type labels yet — those live in the *processed* companion dataset (see §5)

---

## 2. QC: What to Filter and Why

The notebook's QC block computes three per-cell metrics:

```python
adata.obs["library_size"] = adata.X.sum(axis=1)     # total UMIs per cell
adata.obs["n_genes"]      = (adata.X > 0).sum(1)    # genes detected per cell
adata.var["mt"]           = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
```

### What you're filtering against

| Metric | Low → problem | High → problem |
|--------|---------------|----------------|
| `library_size` | Empty droplets / dying cells | Doublets |
| `n_genes` | Broken cell / ambient RNA | Doublets |
| `pct_counts_mt` | — | Stressed or dying cells (mitochondrial leakage) |

### Typical thresholds (what the notebook uses)
```python
sc.pp.filter_cells(adata, min_genes=200)       # drop cells with <200 genes
adata = adata[adata.obs.pct_counts_mt < 20]    # drop cells >20% MT
sc.pp.filter_genes(adata, min_cells=3)         # drop genes in <3 cells
```

### Shape you should see after QC
```
After filtering: ~2638 cells, ~13700 genes
```

You'll lose ~60 cells and drop ~19,000 genes that were never detected. This is normal and expected for PBMC 3k.

### Pitfalls to watch for
- **Don't log1p before computing library size.** The notebook does this correctly (QC on raw `.X`).
- **Don't use `pct_counts_mt > 10`** for PBMC 3k. The v1 chemistry gives noisier MT fractions; 20% is the conventional cutoff. For PBMC 10k (v3), you can tighten to 10-15%.

---

## 3. Library Size: The Most Important Per-Cell Number

For NB/ZINB decoders, `library_size` isn't just a QC metric — it's a **covariate the model needs**. The CVAE_NB decoder outputs *rates*, and the library size scales those rates to counts:

```
count ~ NB(mean = library_size × rate, dispersion = θ)
```

### What the notebook does right
- Computes library size **on raw `.X`**
- **Recomputes** after filtering (important — otherwise you have a stale total from pre-filter genes)
- Keeps it as `adata.obs["library_size"]` for the dataloader to pass to the decoder

### What breaks things
- Computing library size on normalized data → all cells look identical → NB degenerates
- Using HVG-subset counts as library size → throws away information about total transcriptional activity

---

## 4. HVG Selection: The One Place Normalization Is OK

Highly variable genes (HVGs) are the subset of genes that carry biological signal. Selecting HVGs is standard before training generative models — it reduces dimensionality from ~13k to ~2k and removes housekeeping genes that mostly add noise.

The notebook's approach:

```python
# Store raw counts BEFORE normalization
adata.layers["counts"] = adata.X.copy()

# HVG selection uses a *temporary* normalized copy
adata_norm = adata.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)
sc.pp.highly_variable_genes(adata_norm, n_top_genes=2000)

# Transfer the HVG flags back; keep the real counts
adata.var["highly_variable"] = adata_norm.var["highly_variable"]
adata = adata[:, adata.var["highly_variable"]].copy()
```

### Why the dance with a copy?
HVG selection is *descriptive* (which genes vary most?) and benefits from variance stabilization via log1p. But we must **not** let that log1p contaminate the actual training data. The copy-then-transfer pattern ensures:
- HVG flags are computed on log-normalized data (statistically sensible)
- Training data remains raw counts (generatively correct)

This is the reason PBMC.md insists on "keep raw counts" — the notebook does the *right* thing by using normalization only for gene selection, not for training.

### Shape after HVG selection
```
Final shape before labels: ~2638 cells × 2000 genes
```

---

## 5. Getting Cell Type Labels

PBMC 3k raw doesn't come with cell type annotations. The scanpy ecosystem ships a **separately processed** version:

```python
adata_processed = sc.datasets.pbmc3k_processed()
```

This gives you Louvain cluster labels mapped to canonical PBMC cell types:

| Louvain cluster | Cell type | Approx. count |
|-----------------|-----------|---------------|
| 0 | CD4 T cells | ~1100 |
| 1 | CD14+ Monocytes | ~480 |
| 2 | B cells | ~340 |
| 3 | CD8 T cells | ~310 |
| 4 | NK cells | ~150 |
| 5 | FCGR3A+ Monocytes | ~160 |
| 6 | Dendritic cells | ~40 |
| 7 | Megakaryocytes | ~15 |

(Exact counts depend on your filtering thresholds.)

The notebook matches cells by barcode:

```python
common_cells = adata.obs_names.intersection(adata_processed.obs_names)
adata = adata[common_cells].copy()
adata.obs["cell_type"] = adata_processed.obs.loc[common_cells, "louvain"].astype("category")
```

**Expect a small loss** (~10-40 cells) because the processed version uses slightly different QC and may drop a few cells yours keeps (or vice versa). The intersection is always safe.

### Why "louvain" as cell type?
These aren't wet-lab ground truth — they're **clusters + canonical marker gene mapping**. Good enough for tutorial purposes. For more rigorous labels on larger datasets, use **Azimuth** reference mapping (§7).

---

## 6. What PBMC 3k Actually Teaches Your Model

With ~2600 cells × 2000 genes, PBMC 3k is tiny by modern scRNA-seq standards. It's the right size for:

- **Fast iteration loops** — full training run in <5 min on a laptop CPU
- **Debugging the CVAE_NB pipeline** — does the loss decrease? does the latent space separate cell types?
- **Sanity-checking count-aware decoders** — NB must beat Gaussian on this data; if it doesn't, something is wrong
- **UMAP visualization** — 8 cell types separate cleanly enough that you can eyeball whether the latent is learning biology

It's **too small** for:
- Meaningful dispersion estimation per gene (you'll see noisy `θ`)
- Rare cell type recovery (megakaryocytes at n=15 are a lottery)
- Batch effect experiments (single donor)
- Anything diffusion-related (density estimation needs more samples)

When PBMC 3k stops surprising you, graduate to one of the datasets below.

---

## 7. Scaling Up: More Complete PBMC Datasets

### 7.1 PBMC 10k (10x Genomics, v3 chemistry)
- **Size**: ~10,000 cells
- **Chemistry**: 10x Chromium v3 — cleaner, higher per-cell UMI counts
- **Why use it**: Same biology as PBMC 3k but 4x the cells and noticeably less technical noise
- **Access**: Not bundled with scanpy. Download from [10x Genomics datasets](https://www.10xgenomics.com/resources/datasets) (search "10k PBMCs v3")
- **When to use**: First step up from PBMC 3k. Validates that your pipeline scales without changing anything conceptually.

### 7.2 PBMC 68k (Zheng et al. 2017)
- **Size**: 68,579 cells
- **Chemistry**: 10x Chromium v1
- **Reference**: Zheng et al., *Nature Communications* 2017 — the foundational large-scale PBMC study
- **Why use it**: Enough data to expose real generative-modeling questions — posterior collapse, latent dimensionality, rare cell type learning, dispersion estimation quality
- **Access**: Available via 10x Genomics, also hosted on scPerturb / CellxGene
- **When to use**: Once PBMC 3k "just works." This is where you benchmark whether your CVAE_NB genuinely beats scGen on a non-toy scale.

### 7.3 Azimuth PBMC Reference (Hao et al. 2021)
- **Size**: ~161,000 cells
- **Modality**: CITE-seq (RNA + 228 surface proteins) — the dual measurement matters
- **Reference**: Hao et al., *Cell* 2021 — the "Level 1/2/3" cell type hierarchy reference
- **Why use it**:
  - **Gold-standard cell type labels** (manually curated, hierarchical)
  - Multi-donor — finally real batch effect testing
  - Protein data lets you validate RNA-based predictions
- **Access**: [SeuratData / Azimuth](https://azimuth.hubmapconsortium.org/references/human_pbmc/), also on CellxGene
- **When to use**: When label quality matters — benchmarking cell type classification from latent space, testing conditional generation with real hierarchical labels.

### 7.4 Stoeckius CITE-seq PBMC
- **Size**: ~8,500 cells
- **Modality**: CITE-seq (RNA + 10 antibodies) — simpler than Azimuth, good starter CITE-seq
- **Reference**: Stoeckius et al., *Nature Methods* 2017 — the paper that introduced CITE-seq
- **When to use**: If you want to extend the CVAE_NB into multimodal territory (joint RNA+protein VAE). Protein counts are also NB-distributed, which keeps the likelihood story consistent.

### 7.5 Lupus PBMC Atlas (Perez et al. 2022)
- **Size**: ~1.2 million cells
- **Donors**: 260 (SLE patients + controls)
- **Reference**: Perez et al., *Science* 2022
- **Why use it**: Disease vs healthy + massive donor variation = a real population-genetics-of-cells problem
- **When to use**: For serious batch correction experiments and conditional generation across disease states. Probably overkill for genai-lab's current scope — but a benchmark target if the perturbation flagship succeeds.

### 7.6 COVID PBMC studies
- **Representative**: Stephenson et al. 2021 (~780k PBMCs), Su et al. 2020
- **Why relevant**: Perturbation-like structure (healthy vs mild vs severe) using only natural biological states
- **When to use**: If disease-state conditioning interests you more than CRISPR perturbations for the flagship.

### Decision guide

| Your question | Dataset |
|---------------|---------|
| Does my CVAE_NB pipeline even work? | **PBMC 3k** |
| Does it still work on modern v3 chemistry? | **PBMC 10k** |
| Does it scale statistically? | **PBMC 68k** |
| Do my cell type labels hold up against gold standard? | **Azimuth reference** |
| Can I handle multimodality? | **Stoeckius CITE-seq** or **Azimuth** |
| Can I handle donor/batch variation at scale? | **Lupus atlas** |
| Do I want disease-state conditioning? | **Lupus atlas** or **COVID studies** |

---

## 8. How PBMC Fits the genai-lab Flagship

The flagship application ([`docs/applications/perturbation_prediction.md`](../../../docs/applications/perturbation_prediction.md)) targets the **Norman et al. 2019 Perturb-seq** dataset — which is **not PBMC**. Norman uses K562 cells (chronic myeloid leukemia cell line) with CRISPR knockouts.

So why spend time on PBMC at all?

1. **The preprocessing stack is identical.** Raw counts → QC → HVG → NB decoder. Everything you learn on PBMC transfers directly to Norman with one change: the conditioning variable (cell type → perturbation label).

2. **PBMC has "natural" conditions; Norman has "engineered" conditions.** Cell types in PBMC play the same role perturbations play in Norman — both are categorical conditioning variables the CVAE_NB needs to learn to generate conditionally on.

3. **Debugging is faster on PBMC.** If your NB decoder isn't learning cleanly on PBMC 3k, don't even bother running it on Norman — you'll burn 10x more training time to discover the same bug.

4. **PBMC is the published sanity check for every scRNA-seq generative method.** scVI, scGen, CPA, scPPDM all demonstrate on PBMC first. Keeping a PBMC baseline lets you compare fairly.

**Practical pipeline:**
```
PBMC 3k   → debug the CVAE_NB, verify NB beats Gaussian      (done in the current notebook)
PBMC 10k  → validate scaling without chemistry weirdness     (optional intermediate)
Norman    → flagship — swap cell_type for perturbation_label (Week 1-2 of the sprint)
```

---

## 9. Quick Reference: Expected Shapes at Each Step

```
sc.datasets.pbmc3k()           → (2700, 32738)  raw counts, v1 chemistry
after var_names_make_unique()  → (2700, 32738)  dedupes gene symbols
after cell filter (min_genes)  → (~2638, 32738)
after MT filter                → (~2638, 32738)
after gene filter (min_cells)  → (~2638, ~13700)
after HVG selection (n=2000)   → (~2638, 2000)  ← training input
after barcode intersection     → (~2600, 2000)  ← with cell_type labels
```

Use these as landmarks. If your shapes diverge by more than ~20 cells from these, re-check your filter thresholds.

---

## 10. References

- **PBMC 3k**: 10x Genomics, 2016. [Demo dataset page](https://www.10xgenomics.com/resources/datasets/2-7-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0)
- **PBMC 68k**: Zheng et al., *Nature Communications* 2017. "Massively parallel digital transcriptional profiling of single cells."
- **Azimuth PBMC**: Hao et al., *Cell* 2021. "Integrated analysis of multimodal single-cell data."
- **CITE-seq**: Stoeckius et al., *Nature Methods* 2017. "Simultaneous epitope and transcriptome measurement in single cells."
- **Lupus atlas**: Perez et al., *Science* 2022. "Single-cell RNA-seq reveals cell type-specific molecular and genetic associations to lupus."
- **Scanpy docs on PBMC**: [scanpy.datasets.pbmc3k](https://scanpy.readthedocs.io/en/stable/api/scanpy.datasets.pbmc3k.html)

---

## Related Documents in genai-lab

- [`docs/datasets/gene_expression/PBMC.md`](../../../docs/datasets/gene_expression/PBMC.md) — theoretical/pedagogical view (why PBMC, why raw counts)
- [`docs/datasets/gene_expression/data_preparation.md`](../../../docs/datasets/gene_expression/data_preparation.md) — preprocessing conventions
- [`docs/VAE/VAE-07-NB-ZINB.md`](../../../docs/VAE/VAE-07-NB-ZINB.md) and [`VAE-08-NB-likelihood.md`](../../../docs/VAE/VAE-08-NB-likelihood.md) — NB/ZINB theory the notebook implements
- [`docs/applications/perturbation_prediction.md`](../../../docs/applications/perturbation_prediction.md) — where PBMC-style preprocessing meets the flagship application
