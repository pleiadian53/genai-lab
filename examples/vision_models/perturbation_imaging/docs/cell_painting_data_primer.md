# Cell Painting Data Primer

Background reading before V1. Covers what cell-painting data actually is,
why it has the structure it has, and what that means for downstream
generative modeling.

---

## What is cell painting?

**Cell painting** is a high-content, image-based screening assay
introduced by Bray et al. (2016) in the Carpenter lab at the Broad
Institute. It is the dominant phenotypic-profiling protocol in
industrial drug screening today.

The basic recipe:

1. Plate cells (typically U2OS, A549, or HUVEC) in 384-well plates.
2. Apply perturbations: small molecules at varying doses, CRISPR
   knockouts, ORF over-expression — one perturbation per well.
3. Stain the cells with a six-dye cocktail (see below) that paints
   five "channels" of cellular structure.
4. Image each well with an automated fluorescence microscope at
   multiple sites (positions within the well).
5. Extract per-cell features (CellProfiler) or use the raw images for
   deep learning.

The output is millions of multi-channel microscopy images, each tagged
with the perturbation that produced it. **The premise: structurally
similar cells (under the assay's stains) suggest mechanistically
similar perturbations.**

This makes cell painting an unsupervised mechanism-of-action discovery
tool — and a natural target for generative modeling, since the *image*
is the readout.

---

## The five-channel staining protocol

Cell painting is named for the five distinguishable channels its dye
cocktail produces, each lighting up a different cellular compartment:

| Channel | Stain(s) | What it reveals |
|---------|----------|-----------------|
| **DNA** | Hoechst 33342 | Nucleus shape, DNA density, mitotic stage |
| **ER** | Concanavalin A (Alexa 488) | Endoplasmic reticulum |
| **RNA** | SYTO 14 | Nucleoli + cytoplasmic RNA |
| **AGP** | Phalloidin (Alexa 568) + WGA | Actin (cytoskeleton), Golgi, plasma membrane |
| **Mito** | MitoTracker Deep Red | Mitochondria |

Channel order in JUMP-CP files follows this convention. Imaging
hardware acquires each channel sequentially using narrow-band
excitation/emission filters; the resulting per-channel intensity
images are the raw data. **A "cell painting image" is therefore a
5-channel tensor**, not an RGB image. Standard pretrained vision
models built for natural-image RGB cannot ingest this directly —
this is the data-driven reason V2 trains a domain-specific multi-channel
VAE rather than reusing SD-VAE.

Some protocols (RxRx1) split AGP into separate phalloidin + WGA
channels, yielding **6 channels** instead of 5.

---

## Image structure: plate → well → site → channel

Cell painting data is hierarchically organized. Each layer matters for
correctness:

```
Experiment / Batch
└── Plate (384 wells, but often 96 wells used)
    └── Well (one perturbation per well)
        └── Site (multiple imaged positions per well, typically 6–9)
            └── Channel (5 or 6 multi-channel TIFFs per site)
```

| Level | Why it matters |
|-------|----------------|
| **Plate** | Pipetting, fixation, and acquisition happen at plate granularity. *Plate effects* (illumination drift, focus, edge artifacts) are the dominant batch effect. |
| **Well** | One perturbation per well. Replicate wells across plates allow within-perturbation variance estimation. |
| **Site** | Sub-sampling within a well. Cells at well edges differ from center cells (meniscus, edge curvature). |
| **Channel** | Each channel has its own intensity scale; cross-channel correlations encode the biology. |

**Implication for splits**: a naive image-level random split leaks
plate effects into the training set. Best practice is **plate-level**
or **batch-level** splits, with held-out plates from the same
experiment for validation. RxRx1 publishes the canonical splits;
JUMP-CP requires the user to construct them.

---

## Public datasets

### RxRx1 (Recursion, 2019)

- **Size**: ~125,000 images, 6 channels
- **Cells**: 4 cell types (HUVEC, RPE, HepG2, U2OS)
- **Perturbations**: 1,108 different siRNAs + non-targeting controls
- **Image size**: 512×512 pixels (raw); often resized to 256×256 for ML
- **Splits**: published train/val/test by experiment, with held-out
  experiments for true generalization
- **Access**: <https://www.rxrx.ai/rxrx1>; CC BY-NC-SA 4.0 license
- **Note**: 6 channels here are Hoechst, ConA, phalloidin, syto14,
  MitoTracker, WGA — phalloidin and WGA are split, unlike JUMP-CP

**Why it's the V1 default**: smaller than JUMP-CP, well-curated, has
canonical splits, and the task (siRNA classification) is non-trivial
but tractable. Tutorial-friendly.

### JUMP-CP (Joint Undertaking for Morphological Profiling, 2023)

- **Size**: ~3,000,000 images across the consortium dataset; the
  "cpg0016-jump" subset is the most commonly benchmarked
- **Cells**: A549, U2OS (also other lines in the broader consortium)
- **Perturbations**: ~116,000 unique perturbations spanning compounds,
  CRISPR knockouts, and ORF over-expressions
- **Image size**: 1080×1080 (varies by source plate)
- **Channels**: 5 (standard cell painting protocol)
- **Access**: AWS S3 bucket `cellpainting-gallery`; CC0 license; large
  enough to require multi-TB local storage or streaming
- **Splits**: not pre-defined — users construct them

**Why it's deferred**: the size and preprocessing complexity make it a
poor V1 starter. A small JUMP-CP subset becomes useful at V5 for
cross-dataset benchmarking.

### BBBC datasets (Broad Bioimage Benchmark Collection)

A library of smaller, curated cell-imaging datasets — useful for
sanity-checking pipelines but lacking the large perturbation panels
that make cell painting interesting for generative modeling. Skip for
this path.

---

## Standard preprocessing pipeline

Raw cell-painting images are noisy in ways that aren't intrinsic to
the biology. A standard QC pipeline (V1 milestone) corrects:

### 1. Illumination correction

Optical illumination across the field of view is non-uniform —
center-of-field cells appear brighter than edge cells. Standard
correction: estimate a per-channel illumination function from a
plate average, then divide each image by it.

- Tool: **pyBaSiC** (Python port of BaSiC, Peng et al. 2017)
- Output: illumination-corrected single-channel TIFFs

### 2. Background subtraction

Each channel has a non-zero baseline from autofluorescence, dye
bleed-through, and camera dark current. Subtract a per-channel
plate-level background estimate (typically the median of empty wells).

### 3. Plate normalization

Different plates have different exposure times, dye lots, and
acquisition parameters. Plate normalization rescales per-channel
intensity distributions to a common reference (typically z-scoring
per channel using DMSO control wells as the reference).

### 4. Compression / segmentation (optional)

For most generative modeling work, full-image tensors are kept and
fed downstream. Some pipelines additionally segment cells out and
work on per-cell crops; this loses spatial context but reduces
dimensionality.

### 5. Batch-effect correction (advanced)

After per-plate normalization, residual batch effects across
*experiments* may remain. Cross-experiment correction (e.g., ComBat,
sphering, or harmony-style) is sometimes applied. RxRx1's published
splits avoid this by holding out *experiments*; JUMP-CP requires
explicit handling.

---

## Tooling

| Tool | Purpose | When you'd use it |
|------|---------|-------------------|
| **pycytominer** | Profile aggregation, normalization, feature selection | Standard for tabular per-cell features; we'll use its normalization helpers in V1 |
| **CellProfiler** | Image-based feature extraction (per-cell morphology features) | Generates the classical 1k+ morphology-feature CSV. Useful for evaluation in V5 (do synthetic samples cluster by perturbation in CellProfiler-feature space?), not for training |
| **pyBaSiC** | Illumination correction | V1 |
| **deepprofiler** | Tile loading, augmentation for deep learning | Optional V1 helper |
| **plate_qc** (in pycytominer) | Empty-well, low-cell-count QC | V1 |

For V1, **pycytominer + pyBaSiC + the published RxRx1 PyTorch loader**
is the minimum viable stack. Adding CellProfiler is V5 work for
evaluation, not training.

---

## What this means for generative modeling

The structure of cell-painting data shapes several design choices in
the V-series:

1. **Multi-channel native** — VAEs (V2), DiT (V3+), and conditioning
   modules all operate on 5- or 6-channel tensors, not RGB. SD-VAE is
   not compatible.
2. **Plate-aware splits** — V1 must use plate-level or experiment-level
   splits, never random. Otherwise, V5 evaluation looks artificially
   good because plate effects have leaked into training.
3. **Perturbation imbalance** — many perturbations have only a few
   replicate wells; some have hundreds (DMSO controls). Sampling
   strategy in V4 must account for this — class-balanced sampling, or
   weighting by inverse replicate count, are the usual approaches.
4. **Replicate variance is the noise floor** — even identical wells
   look different. Generative samples should cluster *near*, not *on
   top of*, real samples in latent space. V5 evaluation reflects this:
   a perfect generator would produce samples whose intra-perturbation
   variance matches real replicate variance.
5. **Channel correlation encodes biology** — the diagnostic signal of
   a perturbation is often in *cross-channel* relationships (e.g.,
   "actin reorganized while mitochondria fragmented"), not in any
   single channel. Multi-channel VAEs and DiT models must preserve
   these cross-channel correlations, not just per-channel statistics.

---

## Recommended starting point for V1

| Choice | Default | Reason |
|--------|---------|--------|
| Dataset | **RxRx1** | Smaller, well-curated, canonical splits |
| Subset | 1 cell type (HUVEC), all 1,108 perturbations + controls | Avoids cell-line as a confounder for the first pass |
| Image size | 256×256 (resize from 512×512 raw) | Tractable on A40; preserves enough morphology |
| Channels | All 6 (RxRx1's split AGP convention) | No reason to drop |
| Splits | RxRx1's published experiment-held-out splits | Use the canonical splits to benchmark against published RxRx1 numbers |
| QC | Illumination correction (pyBaSiC) + plate-level normalization | Minimum viable; add background subtraction if reconstruction quality is poor |

V1 produces this dataset on disk in HDF5 or zarr, with manifest CSVs
mapping images to perturbations + plates + replicates.

---

## References

- Bray et al. (2016), "Cell Painting, a high-content image-based assay
  for morphological profiling using multiplexed fluorescent dyes,"
  *Nature Protocols* 11, 1757–1774
- Chandrasekaran et al. (2023), "JUMP Cell Painting dataset:
  morphological impact of 136,000 chemical and genetic perturbations,"
  *bioRxiv* 2023.03.23.534023
- Peng et al. (2017), "A BaSiC tool for background and shading
  correction of optical microscopy images," *Nature Communications* 8,
  14836
- Recursion Pharmaceuticals (2019), "RxRx1: An image set for cellular
  morphological variation across many experimental batches"
- Way et al. (2022), "pycytominer: Sustainable production of single-cell
  morphological profiles from microscopy images" (software)
