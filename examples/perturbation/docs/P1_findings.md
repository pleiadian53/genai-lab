# P1 Findings — Norman 2019 Download, QC, and Splits

**Milestone**: P1 (`examples/perturbation/P1_download_and_qc.py`)
**Run date**: 2026-06-01
**Source**: scPerturb `NormanWeissman2019_filtered.h5ad` (Zenodo record
13350497, v1.4, 698.7 MB)
**Artifact**: [`results/P1_summary.json`](../results/P1_summary.json)
**Status**: ✅ pipeline runs end-to-end on real data

---

## Headline numbers

| Quantity | Initial | After QC | Dropped |
|----------|--------:|---------:|--------:|
| Cells | 111,445 | 111,391 | 54 (0.05%) |
| Genes | 33,694 | 22,608 | 11,086 (33%) |
| Perturbations | 237 | 237 | 0 |

- **Control cells**: 11,849 (label `control`, 10.6% of cells)
- **Splits** (stratified by perturbation, seed 42, 70/15/15):
  train 77,957 · val 16,717 · holdout 16,717
- **Library size** (raw UMI counts, full filtered gene set):
  median 13,859 · mean 14,679 · range 3,679–64,352 · std 6,011

## What the QC actually did

- **Cell filter** dropped only 54 cells, all on `pct_counts_mt ≥ 20` (range
  0.04–42.63%). **Zero** cells failed `min_genes < 200` — scPerturb's
  `_filtered` release is already cell-QC'd, so our cell thresholds are a
  light safety net, not the primary filter.
- **Gene filter** removed 11,086 genes (33%) on `min_cells < 3` — the bulk of
  the reduction. 22,608 genes remain.
- **Perturbation filter** removed nothing: all 237 perturbations clear the
  30-cell minimum (smallest groups are still well above threshold). The
  control-exempt logic was therefore not exercised this run.
- **Library size** was computed from `layers["counts"]` on the 22,608-gene
  filtered set and stored in `obs["library_size"]` for NB/ZINB decoders.

## Schema confirmed against scPerturb

These were *probed* defensively in the loader; the run confirms the actuals:

| Field | Resolved value |
|-------|----------------|
| Perturbation column | `perturbation` (first candidate) |
| Control label | `control` (first candidate) |
| Combination delimiter | **`_`** (underscore), e.g. `CEBPE_RUNX1T1`, `TBX3_TBX2` |

> ⚠️ **Delimiter correction.** Earlier docs
> ([`docs/JEPA/04_jepa_perturbseq.md`](../../../docs/JEPA/04_jepa_perturbseq.md),
> the dataset tutorial, and the GP-JEPA spec) assumed `+`-separated pairs
> (`MAPK1+BRAF`). scPerturb uses **`_`**. Any combination-parsing code must
> split on `_`, not `+`. Filed for P2 (perturbation encoder) and the
> `combination` split strategy.

## Top perturbations by cell count

| Perturbation | Cells | Type |
|--------------|------:|------|
| control | 11,849 | control |
| KLF1 | 1,959 | single |
| BAK1 | 1,457 | single |
| CEBPE | 1,233 | single |
| CEBPE_RUNX1T1 | 1,217 | **pair** |
| UBASH3B | 1,202 | single |
| ETS2 | 1,201 | single |
| TBX3_TBX2 | 1,167 | **pair** |
| OSR2 | 1,003 | single |
| SLC4A1 | 999 | single |
| SET | 986 | single |

237 perturbation labels total = control + singletons + pairs. (The full
singleton/pair breakdown is a P2 input; see "Open items" below.)

## Corrections to prior estimates

The methodology docs predated any real run and under-counted:

| Doc claim | Actual | Source to fix |
|-----------|--------|---------------|
| "101 genes, single + double" | **237 perturbation labels** | `04_jepa_perturbseq.md` §1.1 |
| "~100K cells" | **111,445** (111,391 post-QC) | `04_jepa_perturbseq.md`, tutorial |
| "5K HVG genes" (load-time) | 22,608 after gene QC; HVG is P2's choice | tutorial expected-shapes |
| Pairs as `A+B` | pairs as `A_B` | tutorial §2, `04` doc, GP-JEPA spec |

## Performance & storage notes

- **QC wall-clock: ~17 min** (1,013 s), dominated by `sc.pp.filter_genes` on
  the 111k × 33.7k sparse matrix (≈10 min) and the `pct_mt` mask `.copy()`
  (≈6.5 min). This is a CSR row-vs-column filtering cost, not a logic
  problem. It is a **one-time** cost: P2–P5 load the QC'd artifact and never
  re-run QC. Load itself is fast (9 s).
- **`NormanWeissman2019_qcd.h5ad` is 5.8 GB** — ~8× the 699 MB source. Two
  causes: (1) written uncompressed, and (2) at P1 `X` and `layers["counts"]`
  are **identical** (no normalization happens in P1), so the count matrix is
  stored twice. Adding `compression="gzip"` on write — and not duplicating
  the counts layer until `X` is actually normalized — would bring this back
  toward ~1 GB. Tracked as a follow-up fix (see Open items).

## Open items → P2

1. **Combination split semantics.** Now that label structure is visible (237
   labels, `_` delimiter), define `split="perturbation"` (hold out whole
   singleton perturbations) and `split="combination"` (train on singletons +
   a subset of pairs, hold out unseen pairs — the genetic-interaction
   generalization test). Implement in
   [`norman.py`](../../../src/genailab/applications/perturbation/data/norman.py)
   `split_norman`, replacing the current `NotImplementedError`.
2. **Singleton vs pair counts.** Compute the exact split (how many of the 237
   are pairs) — drives the `combination` holdout design and the P2 benchmark
   protocol.
3. **Storage fix.** gzip-compress the QC'd write; drop the redundant counts
   layer until normalization occurs.
4. **Perturbation encoder.** Split combinations on `_`; compose pair
   embeddings (mean or learned) per the GP-JEPA / `04` doc design.
</content>
