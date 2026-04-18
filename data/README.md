# Data Directory

Expression datasets, reference annotations, and model checkpoints for
training generative models in genai-lab.

## Not in Git

Only this `README.md` and the local `.gitignore` are tracked. All dataset
files (`.h5ad`, `.h5`, `.npz`, `.parquet`, `.pt`, ...) are excluded.

Large datasets belong on RunPod network volumes — stage them once via
[`ops/provision_cluster.py --stage-data`](../ops/README.md). Do not commit.

## Layout

Organize as `data/<modality>/<sub-topic>/<dataset>/`:

```
data/
├── scrna/                              # Single-cell RNA-seq
│   ├── perturb_seq/                    #   Perturbation response (CRISPR, drug)
│   │   ├── norman_2019/                #     K562 CRISPR screen (flagship)
│   │   └── replogle_2022/              #     Essential-gene screen
│   ├── pbmc/                           #   PBMC atlases
│   │   ├── pbmc3k/                     #     10x v1 demo (~2.7k cells)
│   │   ├── pbmc10k/                    #     10x v3 (~10k cells)
│   │   ├── pbmc68k/                    #     Zheng 2017 (~68k cells)
│   │   └── azimuth/                    #     Hao 2021 reference (~160k CITE-seq)
│   └── tabula_sapiens/                 #   Cross-tissue atlas
├── bulk/                               # Bulk RNA-seq
│   ├── gtex/                           #   Tissue-level expression
│   ├── recount3/                       #   Uniformly re-processed public RNA-seq
│   └── tcga/                           #   Tumor atlases
└── models/                             # Trained model checkpoints
    ├── baseline_cvae_nb/
    └── jepa_perturb/
```

**Why modality first?** Data loaders, preprocessing steps, and likelihood
choices all depend on modality (scRNA-seq → raw counts + NB decoder; bulk →
log-normalized + Gaussian decoder). Putting modality at the top level makes
the path self-explanatory and keeps modality-specific tooling easy to route.

**New modalities go at the top level.** If genai-lab adds medical imaging or
protein structure work later, add `data/imaging/` or `data/structure/`
siblings — don't nest them under an existing modality.

## Accessing Data from Code

```python
from genailab.data.paths import get_data_paths

paths = get_data_paths()
pbmc3k = paths.scrna_processed("pbmc3k")                     # single-cell convenience
norman = paths.scrna_processed("perturb_seq/norman_2019")    # sub-topic path
gtex   = paths.bulk_processed("gtex")                        # bulk modality
```

See [`src/genailab/data/paths.py`](../src/genailab/data/paths.py) for the full
path resolution logic.

## Staging to RunPod

Datasets are too large to re-upload on every training run. Stage each one
to the RunPod network volume once:

```bash
python ops/provision_cluster.py --stage-data --data-path scrna/perturb_seq/norman_2019
python ops/provision_cluster.py --stage-data --data-path scrna/pbmc/68k
python ops/provision_cluster.py --stage-data --data-path bulk/gtex
```

Subsequent provisions mount the volume instantly. See
[`ops/README.md`](../ops/README.md) for the full workflow.
