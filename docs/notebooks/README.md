# GenAI Lab Interactive Notebooks

Interactive Jupyter notebooks demonstrating generative AI models for computational biology.

---

## 📂 Notebook Organization

Notebooks are organized by topic:

- **`diffusion/`** - DDPM, DiT, latent diffusion tutorials
- **`vae/`** - VAE, β-VAE, CVAE examples
- **`foundation_models/`** - Fine-tuning and adaptation
- **`datasets/`** - Data loading and preprocessing

---

## 📍 Viewing Options

### **Option 1: GitHub Pages (Best Rendering)** ⭐ **Recommended**

View rendered notebooks with proper math and plots:

**Diffusion Models:**
- 📊 [DDPM Basics](https://pleiadian53.github.io/genai-lab/diffusion/01_ddpm/01_ddpm_basics/)
- 📊 [SDE Formulation](https://pleiadian53.github.io/genai-lab/diffusion/02_sde_formulation/02_sde_formulation/)
- 📊 [Medical Imaging Diffusion](https://pleiadian53.github.io/genai-lab/diffusion/03_medical_imaging_diffusion/03_medical_imaging_diffusion/)
- 📊 [Gene Expression Diffusion](https://pleiadian53.github.io/genai-lab/diffusion/04_gene_expression_diffusion/04_gene_expression_diffusion/)

**VAE Series:**
- 📋 Coming soon

**Advantages:**

- ✅ Reliable rendering (no GitHub timeouts)
- ✅ Math properly displayed  
- ✅ Plots and outputs preserved
- ✅ Mobile-friendly

### **Option 2: Run Locally (Interactive)**

Clone and run in Jupyter:

```bash
# Clone repository
git clone https://github.com/pleiadian53/genai-lab.git
cd genai-lab/notebooks  # Use the original location

# Create environment
conda env create -f ../environment.yml
conda activate genai-lab

# Or install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

**Advantages:**

- ✅ Fully interactive
- ✅ Modify and experiment
- ✅ Run with your own data

### **Option 3: GitHub.com (Quick View)**

Browse notebooks directly on GitHub:
- 📓 [View on GitHub](https://github.com/pleiadian53/genai-lab/tree/main/notebooks)

**Note:** GitHub's notebook renderer can be slow/unreliable for large notebooks. Use GitHub Pages for best experience.

---

## 📝 Note for Contributors

These notebooks are **rendered copies** from the main `/notebooks/` directory in the repository.

### **Development Workflow:**

1. **Develop** in `/notebooks/` (primary location)
   - Experiment freely
   - Iterate on code and outputs
   - Keep messy/experimental notebooks here

2. **Polish** before publishing
   - Clear outputs or save clean outputs
   - Add markdown explanations
   - Test end-to-end execution

3. **Publish** to `/docs/notebooks/` (rendered location)
   - Copy notebook when ready for public viewing:
     ```bash
     cp notebooks/diffusion/my_tutorial.ipynb docs/notebooks/diffusion/
     ```
   - Add to git and push

4. **Add to navigation** (optional)
   - Edit `mkdocs.yml` to include in nav structure
   - Notebook is accessible by URL even without nav entry

### **Why Two Locations?**

- **`/notebooks/`** — Source of truth, standard location for developers
- **`/docs/notebooks/`** — Rendered version for documentation site

This dual-location approach:
- ✅ Keeps notebooks where developers expect them
- ✅ Enables reliable rendering on GitHub Pages
- ✅ Allows selective publishing (not all notebooks need to be public)
- ✅ Explicit "publish" step ensures quality

**Duplication is intentional and minimal** - only polished notebooks are copied.

---

## 🚀 Available Notebooks

### **Diffusion Models** ✅ Available Now!

| Notebook | Description | Link | Status |
|----------|-------------|------|--------|
| **DDPM Basics** | Introduction to Denoising Diffusion Probabilistic Models | `notebooks/diffusion/01_ddpm/` | ✅ Available |
| **SDE Formulation** | Stochastic Differential Equations for diffusion | `notebooks/diffusion/02_sde_formulation/` | ✅ Available |
| **Medical Imaging Diffusion** | Diffusion models for medical images | `notebooks/diffusion/03_medical_imaging_diffusion/` | ✅ Available |
| **Gene Expression Diffusion** | Applying diffusion to gene expression data | `notebooks/diffusion/04_gene_expression_diffusion/` | ✅ Available |

**Quick access:** Navigate to **"Diffusion (Technical Deep Dives)"** in the top navigation tabs or sidebar!

### **VAE Series**

| Notebook | Description | Status |
|----------|-------------|--------|
| Coming soon | VAE training | 📋 Planned |

### **Foundation Models**

| Notebook | Description | Status |
|----------|-------------|--------|
| Coming soon | Fine-tuning guide | 📋 Planned |

---

## 📚 Related Resources

- **[Documentation](https://pleiadian53.github.io/genai-lab/)** — Theory and concepts
- **[Examples](https://github.com/pleiadian53/genai-lab/tree/main/examples)** — Runnable scripts
- **[Source Code](https://github.com/pleiadian53/genai-lab/tree/main/src)** — Implementation

---

## 💡 Tips for Learning

- **Start with documentation** to understand theory
- **Run notebooks** to see models in action
- **Modify and experiment** to deepen understanding
- **Check examples/** for production-ready scripts

---

## 🤝 Contributing

Found a bug? Have suggestions for new notebooks?

1. **Open an issue:** [GitHub Issues](https://github.com/pleiadian53/genai-lab/issues)
2. **Suggest topics:** What models/techniques would help your research?
3. **Share your notebooks:** Built your own? We'd love to feature them!

---

**Ready to start?** Clone the repo and explore `/notebooks/`! 🚀
