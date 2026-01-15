# GenAI-Lab Setup Guide

This guide explains how to properly install and configure the `genailab` package for development and notebook usage.

## Quick Start

### 1. Install the Package in Editable Mode

The `genailab` package uses Poetry for dependency management. Install it in editable mode so notebooks can import it directly without `sys.path` manipulation:

```bash
# Activate the genailab conda environment
mamba activate genailab

# Install the package in editable mode
cd /path/to/genai-lab
poetry install

# Or if you prefer pip
pip install -e .
```

### 2. Verify Installation

```python
# In Python or Jupyter notebook
import genailab
from genailab import get_config, get_data_dir

print(f"GenAI-Lab version: {genailab.__version__}")
print(f"Project root: {get_config().project_root}")
print(f"Data directory: {get_data_dir()}")
```

You should see output like:
```
GenAI-Lab version: 0.1.0
Project root: /Users/yourname/work/genai-lab
Data directory: /Users/yourname/work/genai-lab/data
```

## Configuration System

### Using the Config Module

The `genailab.config` module provides centralized configuration for paths, datasets, and models:

```python
from genailab import get_config, get_data_dir, get_checkpoint_dir, get_device

# Get configuration
config = get_config()

# Access paths
data_dir = get_data_dir()
checkpoint_dir = get_checkpoint_dir("diffusion/medical_imaging")
device = get_device()  # Auto-detects 'cuda', 'mps', or 'cpu'

# Use in your code
data_path = data_dir / "chest_xray" / "images"
checkpoint_path = checkpoint_dir / "model_epoch_1000.pt"
```

### Directory Structure

After installation, the config module automatically creates:

```
genai-lab/
├── data/              # Datasets
├── checkpoints/       # Model checkpoints
│   └── diffusion/
│       ├── medical_imaging/
│       └── gene_expression/
├── results/           # Experiment results
└── .cache/            # Cached data
```

### Registering Datasets

```python
from genailab import get_config

config = get_config()

# Register dataset paths
config.register_dataset("chest_xray", "/path/to/chest_xray_dataset")
config.register_dataset("gene_expression", "/path/to/gene_data.h5")

# Retrieve later
chest_xray_path = config.get_dataset_path("chest_xray")
```

### Model Configuration Presets

```python
from genailab import get_diffusion_config

# Get preset configurations
unet_config = get_diffusion_config("unet2d_medium")
# Returns: {'in_channels': 1, 'out_channels': 1, 'base_channels': 64, ...}

gene_config = get_diffusion_config("tabular_gene_expression")
# Returns: {'hidden_dim': 512, 'num_layers': 8, ...}

# Use in model initialization
from genailab.diffusion import UNet2D
model = UNet2D(**unet_config)
```

Available presets:
- `unet2d_small`: 32 base channels, 3 levels
- `unet2d_medium`: 64 base channels, 4 levels
- `unet2d_large`: 128 base channels, 4 levels
- `tabular_gene_expression`: MLP with attention for gene data

## Notebook Best Practices

### ❌ Old Way (Don't Do This)

```python
# BAD: Hardcoded path manipulation
import sys
from pathlib import Path
sys.path.insert(0, str(Path('../../../src').resolve()))

from genailab.diffusion import VPSDE
```

### ✅ New Way (Recommended)

```python
# GOOD: Direct import after editable install
from genailab.diffusion import VPSDE
from genailab import get_config, get_checkpoint_dir, get_device

# Use configuration
config = get_config()
device = get_device()
checkpoint_dir = get_checkpoint_dir("my_experiment")
```

### Example Notebook Setup Cell

```python
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import genailab modules
from genailab.diffusion import VPSDE, UNet2D, train_score_network
from genailab import get_config, get_checkpoint_dir, get_device

# Configuration
config = get_config()
device = get_device()
checkpoint_dir = get_checkpoint_dir("diffusion/medical_imaging")

print(f"Device: {device}")
print(f"Checkpoints: {checkpoint_dir}")

# Random seeds
np.random.seed(42)
torch.manual_seed(42)
```

## Advanced Configuration

### Custom Configuration

```python
from genailab.config import Config
from pathlib import Path

# Create custom configuration
custom_config = Config(
    project_root=Path("/custom/path"),
    device="cuda"
)

# Register custom datasets
custom_config.register_dataset("my_dataset", "/path/to/data")

# Register custom model configs
custom_config.register_model_config("my_unet", {
    "in_channels": 3,
    "base_channels": 96,
    "channel_multipliers": (1, 2, 4, 8, 16),
})

# Use globally
from genailab.config import set_config
set_config(custom_config)
```

### Environment Variables

You can override paths using environment variables:

```bash
export GENAILAB_DATA_DIR=/mnt/data/genailab
export GENAILAB_CHECKPOINT_DIR=/mnt/checkpoints
```

Then in Python:
```python
import os
from genailab.config import Config

config = Config(
    data_dir=Path(os.getenv("GENAILAB_DATA_DIR", config.data_dir)),
    checkpoint_dir=Path(os.getenv("GENAILAB_CHECKPOINT_DIR", config.checkpoint_dir)),
)
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'genailab'`

**Solution**: Install the package in editable mode:
```bash
cd /path/to/genai-lab
poetry install
# or
pip install -e .
```

**Verify**:
```python
import genailab
print(genailab.__file__)
# Should show: /path/to/genai-lab/src/genailab/__init__.py
```

### Wrong Project Root

**Problem**: Config detects wrong project root

**Solution**: Explicitly set project root:
```python
from genailab.config import Config
from pathlib import Path

config = Config(project_root=Path("/correct/path/to/genai-lab"))
```

### Device Detection Issues

**Problem**: Wrong device detected (e.g., using CPU when GPU available)

**Solution**: Manually set device:
```python
from genailab.config import Config

config = Config(device="cuda")  # or "mps" or "cpu"
```

## Migration Guide

### Updating Existing Notebooks

1. **Remove sys.path manipulation**:
   ```python
   # DELETE these lines
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path('../../../src').resolve()))
   ```

2. **Add configuration imports**:
   ```python
   # ADD these lines
   from genailab import get_config, get_checkpoint_dir, get_device
   ```

3. **Update checkpoint paths**:
   ```python
   # OLD
   checkpoint_dir = './checkpoints'
   
   # NEW
   checkpoint_dir = get_checkpoint_dir("diffusion/medical_imaging")
   ```

4. **Update device detection**:
   ```python
   # OLD
   device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
   
   # NEW
   device = get_device()
   ```

## Summary

**Key Benefits**:

- ✅ No more `sys.path` hacks in notebooks
- ✅ Centralized configuration for paths and settings
- ✅ Auto-detection of project root and device
- ✅ Preset model configurations
- ✅ Easy dataset registration and retrieval
- ✅ Consistent directory structure across experiments

**Next Steps**:
1. Install package: `poetry install`
2. Update notebooks to use config module
3. Register your datasets
4. Use preset model configurations
