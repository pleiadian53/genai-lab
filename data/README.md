# Data Directory

This directory contains expression datasets for training generative models.

## Structure

```
data/
├── scrna/           # scRNA-seq datasets
│   ├── pbmc3k/
│   ├── pbmc68k/
│   └── tabula_sapiens/
├── bulk/            # Bulk RNA-seq datasets
│   ├── gtex/
│   ├── recount3/
│   └── tcga/
└── models/          # Trained model checkpoints
```

## Usage

```python
from genailab.data.paths import get_data_paths

paths = get_data_paths()
pbmc3k = paths.scrna_processed('pbmc3k')
```
