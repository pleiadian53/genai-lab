# genai-lab Setup on RunPods (using A40 as an example)

Complete step-by-step guide for setting up the `genai-lab` environment on a fresh RunPods GPU pod.

**GPU**: NVIDIA A40 (48GB VRAM) - recommended for full training
**Base Image**: Typically Ubuntu with CUDA pre-installed
**Working Directory**: `/workspace/`

---

## Prerequisites

- RunPods account with GPU pod deployed (A40, A100, or similar)
- SSH or web terminal access to the pod
- GitHub access configured (for cloning private repos)

---

## Step 1: Install Miniforge (Mamba + Conda)

Miniforge provides both `mamba` and `conda`. Mamba is a faster drop-in replacement for conda.

```bash
# Navigate to a temporary location
cd /tmp

# Download the latest Miniforge installer
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Run the installer (non-interactive mode)
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3

# Clean up installer
rm Miniforge3-$(uname)-$(uname -m).sh
```

---

## Step 2: Initialize Shell for Conda/Mamba

This step is **critical** — without it, `mamba activate` will fail.

```bash
# Initialize conda for bash
~/miniforge3/bin/conda init bash

# Reload shell configuration
source ~/.bashrc
```

**Verify installation:**

```bash
mamba --version
conda --version
```

You should see version numbers for both.

---

## Step 3: (Optional) Disable Auto-Activation of Base

If you don't want `(base)` to activate automatically on every login:

```bash
conda config --set auto_activate_base false
```

---

## Step 4: Clone the genai-lab Repository

```bash
cd /workspace

# Clone via HTTPS (if no SSH key configured)
git clone https://github.com/YOUR_USERNAME/genai-lab.git

# OR clone via SSH (if SSH key is configured)
git clone git@github.com:YOUR_USERNAME/genai-lab.git

cd genai-lab
```

**Note**: Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Step 5: Create the genai-lab Environment

```bash
cd /workspace/genai-lab

# Create environment from environment.yml
mamba env create -f environment.yml
```

This will:

- Create a conda environment named `genailab`
- Install Python 3.10+
- Install PyTorch with CUDA support
- Install diffusion model dependencies (tqdm, scipy, etc.)
- Install bio dependencies (scanpy, anndata) if specified

**Expected time**: 3-10 minutes depending on network speed.

---

## Step 6: Activate the Environment

```bash
mamba activate genailab
```

> **Note**: If you encounter a shell initialization error, run:
>
> ```bash
> /root/miniforge3/bin/mamba shell init --shell bash --root-prefix=/root/miniforge3
> source ~/.bashrc
> mamba activate genailab
> ```

**Verify activation:**

```bash
# Should show the genailab environment path
which python

# Should show Python 3.10+
python --version

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## Step 7: Install genai-lab in Editable Mode

```bash
cd /workspace/genai-lab

# Install with pip in editable mode
pip install -e .

# Or if using poetry
# poetry install
```

---

## Step 8: Register Jupyter Kernel (for Notebooks)

This allows VSCode and Jupyter to detect the `genailab` environment:

```bash
python -m ipykernel install --user --name genailab --display-name "Python (genailab)"
```

---

## Step 9: Verify Setup

```bash
# Check key imports
python -c "from genailab.diffusion import VPSDE, UNet2D; print('diffusion OK')"
python -c "from genailab import get_config, get_device; print('config OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run a quick test with the driver script
python scripts/diffusion/03_medical_imaging_diffusion.py --preset tiny --epochs 5
```

---

## Quick Reference Commands

| Task                   | Command                                                      |
| ---------------------- | ------------------------------------------------------------ |
| Activate environment   | `mamba activate genailab`                                    |
| Deactivate environment | `mamba deactivate`                                           |
| List environments      | `mamba env list`                                             |
| Update environment     | `mamba env update -f environment.yml`                        |
| Remove environment     | `mamba env remove -n genailab`                               |
| Check GPU              | `nvidia-smi`                                                 |
| Check PyTorch GPU      | `python -c "import torch; print(torch.cuda.is_available())"` |

---

## Running Training Scripts

### Quick Test (verify logic)

```bash
python scripts/diffusion/03_medical_imaging_diffusion.py --preset tiny --epochs 10
```

### Full Training on A40

```bash
python scripts/diffusion/03_medical_imaging_diffusion.py \
    --preset large \
    --epochs 5000 \
    --sample \
    --experiment-name "xray_a40_run1"
```

### Custom Configuration

```bash
python scripts/diffusion/03_medical_imaging_diffusion.py \
    --base-channels 64 \
    --channel-mults 1,2,4,8 \
    --num-res-blocks 2 \
    --img-size 128 \
    --epochs 10000 \
    --batch-size 32 \
    --sample
```

### List Available Presets

```bash
python scripts/diffusion/03_medical_imaging_diffusion.py --list-presets
```

---

## Troubleshooting

### "mamba activate" fails with shell initialization error

Run:

```bash
eval "$(mamba shell hook --shell bash)"
mamba activate genailab
```

Or re-initialize:

```bash
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### VSCode doesn't detect the environment

1. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Enter path: `~/miniforge3/envs/genailab/bin/python`

Or reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"

### CUDA not available in PyTorch

Verify CUDA is installed on the pod:

```bash
nvidia-smi
nvcc --version
```

If PyTorch was installed without CUDA support, reinstall:

```bash
mamba activate genailab
mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Out of Memory (OOM) during training

Reduce model size or batch size:

```bash
# Use smaller preset
python scripts/diffusion/03_medical_imaging_diffusion.py --preset small

# Or reduce batch size
python scripts/diffusion/03_medical_imaging_diffusion.py --preset medium --batch-size 8
```

---

## One-Liner Setup Script

For convenience, save this as `/workspace/setup_genailab.sh`:

```bash
#!/bin/bash
set -e

echo "=== Step 1: Installing Miniforge ==="
cd /tmp
wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3
rm Miniforge3-$(uname)-$(uname -m).sh

echo "=== Step 2: Initializing shell ==="
~/miniforge3/bin/conda init bash
source ~/.bashrc

echo "=== Step 3: Disabling base auto-activation ==="
conda config --set auto_activate_base false

echo "=== Step 4: Cloning genai-lab ==="
cd /workspace
git clone https://github.com/YOUR_USERNAME/genai-lab.git || echo "Repo already exists"
cd genai-lab

echo "=== Step 5: Creating environment ==="
eval "$(mamba shell hook --shell bash)"
mamba env create -f environment.yml

echo "=== Step 6: Activating and installing ==="
mamba activate genailab
pip install -e .

echo "=== Step 7: Registering Jupyter kernel ==="
python -m ipykernel install --user --name genailab --display-name "Python (genailab)"

echo "=== Step 8: Verifying setup ==="
python -c "from genailab.diffusion import VPSDE, UNet2D; print('✓ genailab imports OK')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo "=== Setup Complete ==="
echo "Run 'mamba activate genailab' to activate the environment."
```

Run with: `bash setup_genailab.sh`

---

## Setting Up SSH Connection for GitHub

To use `git push` and `git pull` without entering credentials, configure SSH authentication.

### 1. Generate SSH Key on the Pod

```bash
ssh-keygen -t ed25519 -C "runpod-genailab-key"
```

This creates:
- Private key: `/root/.ssh/id_ed25519`
- Public key: `/root/.ssh/id_ed25519.pub`

### 2. Retrieve the Public Key

```bash
cat /root/.ssh/id_ed25519.pub
```

Copy the entire output.

### 3. Add Key to GitHub

1. Go to [GitHub](https://github.com) → **Settings** → **SSH and GPG keys**
2. Click **New SSH key**
3. Paste the public key and save

### 4. Test Connection

```bash
ssh -T git@github.com
```

### 5. Configure Git Identity

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Uploading Datasets

### Option A: Git LFS (for smaller datasets < 2GB)

```bash
# Install git-lfs if not present
apt-get install git-lfs
git lfs install

# Track large files
git lfs track "*.npy"
git lfs track "*.h5"
git lfs track "data/**"
```

### Option B: rsync/scp (for larger datasets)

From your local machine:

```bash
# Get pod IP and port from RunPods dashboard
rsync -avz --progress ./data/ root@POD_IP:/workspace/genai-lab/data/ -e "ssh -p PORT"
```

### Option C: Cloud Storage (S3, GCS)

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Sync data
aws s3 sync s3://your-bucket/data/ /workspace/genai-lab/data/
```

### Option D: Hugging Face Datasets

```python
from datasets import load_dataset

# Download directly to pod
dataset = load_dataset("your-dataset-name")
```

---

## Model Size Reference

| Preset | Parameters | VRAM (est.) | Recommended GPU |
| ------ | ---------- | ----------- | --------------- |
| tiny   | ~1M        | <2GB        | Any (CPU OK)    |
| small  | ~5M        | ~4GB        | T4, RTX 3060    |
| medium | ~20M       | ~8GB        | RTX 3080, A10   |
| large  | ~50M       | ~16GB       | A40, A100       |

---

## Related Documents

- [INSTALL.md](../../INSTALL.md) - Local installation guide
- [SETUP.md](../SETUP.md) - Detailed configuration documentation

---

## Appendix: Local SSH Configuration

Before connecting to a RunPod instance, configure SSH on your **local machine** for seamless access.

### Step 1: Create or Edit SSH Config

Open `~/.ssh/config` on your local machine and add an entry for genai-lab:

```bash
# RunPod Instance for genai-lab GPU Training
# Created: YYYY-MM-DD
# Instance ID: [copy from RunPods console]
# GPU: [A40 48GB / A100 80GB / RTX 4090 24GB / etc.]
# Purpose: Diffusion model training, gene expression generation
Host runpod-genai
    # ⚠️ UPDATE: Copy IP from "SSH over exposed TCP" section in RunPods console
    # Example: ssh root@69.30.85.30 -p 22084 -i ~/.ssh/id_ed25519 
    HostName 69.30.85.30
    
    # ⚠️ UPDATE: Copy Port from "SSH over exposed TCP" section
    Port 22084
    
    # RunPods instances use root user
    User root
    
    # Use your SSH key (generate with: ssh-keygen -t ed25519)
    IdentityFile ~/.ssh/id_ed25519
    
    # Skip host key checking (RunPods IPs change frequently)
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    
    # Keep connection alive (prevents disconnections during long training)
    ServerAliveInterval 60
    ServerAliveCountMax 5
    
    # Optional: Compression for faster file transfers
    Compression yes
```

### Step 2: Update Config When Pod Restarts

Each time you start a new pod (or restart an existing one), the IP and port may change:

1. Go to RunPods console → Your pod → **Connect** → **SSH over exposed TCP**
2. Copy the new `HostName` (IP) and `Port` values
3. Update your `~/.ssh/config` entry

### Step 3: Connect

```bash
# Simple connection
ssh runpod-genai

# With VS Code Remote SSH
code --remote ssh-remote+runpod-genai /workspace/genai-lab
```

### Step 4: Verify Connection

```bash
# Check GPU
nvidia-smi

# Verify workspace
ls /workspace
```

### Tips

- **Multiple pods**: Create separate entries (e.g., `runpod-genai-a40`, `runpod-genai-a100`)
- **Port forwarding**: Add `LocalForward 8888 localhost:8888` for Jupyter access
- **File sync**: Use `rsync -avz --progress ./data/ runpod-genai:/workspace/genai-lab/data/`
