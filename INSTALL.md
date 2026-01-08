# GenAI-Lab Installation

## Quick Install

```bash
# 1. Activate conda environment
mamba activate genailab

# 2. Install package in editable mode
cd /path/to/genai-lab
poetry install

# 3. Verify installation
python -c "import genailab; print(f'GenAI-Lab {genailab.__version__} installed successfully')"
```

## What This Does

Installing in **editable mode** means:
- ✅ Notebooks can import `genailab` directly
- ✅ No need for `sys.path.insert()` hacks
- ✅ Changes to source code are immediately available
- ✅ Proper package structure and imports

## After Installation

### In Notebooks

```python
# Just import directly!
from genailab.diffusion import VPSDE, UNet2D
from genailab import get_config, get_checkpoint_dir, get_device

# Configuration is automatic
device = get_device()  # Auto-detects cuda/mps/cpu
checkpoint_dir = get_checkpoint_dir("my_experiment")
```

### Directory Structure Created

```
genai-lab/
├── data/              # Your datasets
├── checkpoints/       # Model checkpoints
│   └── diffusion/
│       ├── medical_imaging/
│       └── gene_expression/
├── results/           # Experiment results
└── .cache/            # Cached data
```

## Troubleshooting

**Import Error?**
```bash
# Make sure you're in the right environment
mamba activate genailab

# Reinstall
poetry install --no-cache
```

**Still not working?**
```bash
# Check if package is installed
pip list | grep genailab

# Should show: genailab 0.1.0 /path/to/genai-lab/src
```

## Full Documentation

See [docs/SETUP.md](docs/SETUP.md) for complete configuration options.
