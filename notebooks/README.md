# Notebooks

Educational tutorials and exploratory notebooks for understanding generative AI methods in computational biology.

## Purpose

These notebooks are designed for:
- **Learning** core concepts and theory behind generative models
- **Exploring** mathematical foundations and derivations
- **Visualizing** model behavior and training dynamics
- **Experimenting** with ideas before productionizing

For production-ready scripts and real-world applications, see `../examples/`.

---

## Directory Structure

```
notebooks/
├── foundation_models/  # 🆕 Foundation model adaptation (LoRA, adapters, resource management)
├── diffusion/          # Diffusion models (DDPM, score-based, flow matching)
├── vae/                # Variational autoencoders
└── foundations/        # Mathematical foundations (score functions, energy models)
```

---

## Learning Paths

### Path 1: Foundation Model Adaptation (New!)

**Prerequisites**: Basic PyTorch, understanding of transformers

1. **Resource-Aware Training**
   - `foundation_models/01_model_sizes_and_resources.ipynb` (planned) — Hardware detection and model sizing
   - Learn to configure models for M1 Mac, RunPod, or Cloud GPUs

2. **Parameter-Efficient Fine-Tuning**
   - `foundation_models/02_lora_basics.ipynb` (planned) — LoRA fundamentals
   - `foundation_models/03_adapters_vs_lora.ipynb` (planned) — Comparing strategies
   - `foundation_models/04_freezing_strategies.ipynb` (planned) — Transfer learning

3. **End-to-End Application**
   - `foundation_models/07_end_to_end_gene_expression.ipynb` (planned) — Complete pipeline
   - See also: `docs/foundation_models/` for theory

### Path 2: Generative Models for Gene Expression

**Prerequisites**: Basic probability, PyTorch, gene expression data

1. **VAE Foundations**
   - Start with `vae/01_vae_theory.ipynb` (coming soon)
   - See also: `docs/VAE/` for detailed theory

2. **Diffusion Models**
   - `diffusion/01_ddpm_basics.ipynb` — Core DDPM on gene expression
   - `diffusion/02_score_matching.ipynb` (planned) — Score-based perspective
   - `diffusion/03_flow_matching.ipynb` (planned) — Flow matching alternative

3. **Applications**
   - Move to `examples/` for production workflows

### Path 3: Drug Response Prediction

**Goal**: Predict perturbation effects using diffusion models (scPPDM approach)

1. `diffusion/01_ddpm_basics.ipynb` — Understand DDPM mechanics
2. `diffusion/04_conditional_generation.ipynb` (planned) — Conditional diffusion
3. `foundation_models/05_conditioning_patterns.ipynb` (planned) — FiLM, cross-attention
4. `examples/perturbation/scPPDM_pipeline.py` — Production implementation

### Path 4: Mathematical Foundations

**For researchers developing novel methods**

1. `foundations/01_score_functions.ipynb` (planned) — Score matching theory
2. `foundations/02_energy_models.ipynb` (planned) — Energy-based models
3. See also: `docs/score_matching/`, `docs/EBM/`

---

## Notebook vs Examples

| Feature | `notebooks/` | `examples/` |
|---------|-------------|-------------|
| Format | `.ipynb` (Jupyter) | `.py` (Python scripts) |
| Purpose | Learn & explore | Apply & deploy |
| Style | Step-by-step, verbose | Modular, efficient |
| Execution | Interactive | Command-line |
| Data | Small subsets | Full datasets |
| Code | Exploratory | Production-ready |

---

## Running Notebooks

### Setup

```bash
# Activate environment
mamba activate genailab

# Launch Jupyter
jupyter lab

# Navigate to notebooks/
```

### Data Requirements

Most notebooks use small datasets for fast iteration:
- **PBMC 3k**: ~2.7k cells, 500 HVGs (~5MB)
- **Toy datasets**: Generated on-the-fly

For full-scale experiments, use scripts in `examples/`.

---

## Contributing

When adding new notebooks:

1. **Organize by topic**: Place in appropriate subdirectory
2. **Follow naming convention**: `01_descriptive_name.ipynb`
3. **Include learning objectives**: State what the notebook teaches
4. **Use small data**: Keep runtime under 10 minutes
5. **Add to this README**: Update learning paths

---

## Planned Notebooks

### Foundation Models (New!)
- [ ] `01_model_sizes_and_resources.ipynb` — Hardware detection and resource management
- [ ] `02_lora_basics.ipynb` — LoRA fundamentals and implementation
- [ ] `03_adapters_vs_lora.ipynb` — Comparing adaptation strategies
- [ ] `04_freezing_strategies.ipynb` — Transfer learning patterns
- [ ] `05_conditioning_patterns.ipynb` — FiLM, cross-attention, CFG
- [ ] `06_mixture_of_experts.ipynb` — Advanced architectures
- [ ] `07_end_to_end_gene_expression.ipynb` — Complete pipeline

### Diffusion Models
- [ ] `02_score_matching.ipynb` — Score-based generative models
- [ ] `03_flow_matching.ipynb` — Flow matching and optimal transport
- [ ] `04_conditional_generation.ipynb` — Guidance mechanisms
- [ ] `05_latent_diffusion.ipynb` — Diffusion in VAE latent space

### VAE
- [ ] `01_vae_theory.ipynb` — VAE basics and ELBO
- [ ] `02_beta_vae.ipynb` — Disentanglement
- [ ] `03_vae_for_biology.ipynb` — Count data and biological constraints

### Foundations
- [ ] `01_score_functions.ipynb` — Fisher score, Stein score
- [ ] `02_energy_models.ipynb` — EBMs and Boltzmann distributions
- [ ] `03_langevin_dynamics.ipynb` — Sampling via Langevin MCMC

---

## Related Resources

- **Documentation**: `docs/` for detailed theory and derivations
- **Examples**: `examples/` for production scripts
- **Source code**: `src/genailab/` for reusable modules
