# Foundation Model Adaptation Tutorials

Interactive tutorials for learning parameter-efficient fine-tuning and foundation model adaptation patterns.

## Tutorial Series

### 1. Model Sizes and Resource Management
**Notebook**: `01_model_sizes_and_resources.ipynb`

Learn how to:
- Auto-detect hardware capabilities
- Select appropriate model sizes
- Estimate memory requirements
- Configure gradient accumulation for limited resources

**Best for**: Understanding hardware constraints and model sizing

---

### 2. LoRA Basics
**Notebook**: `02_lora_basics.ipynb`

Learn how to:
- Apply LoRA to transformer models
- Train only 1% of parameters
- Save and load LoRA weights
- Compare LoRA vs full fine-tuning

**Best for**: Parameter-efficient fine-tuning

---

### 3. Adapters vs LoRA
**Notebook**: `03_adapters_vs_lora.ipynb`

Learn how to:
- Implement adapter modules
- Compare adapters and LoRA
- Choose the right strategy for your task
- Benchmark memory and performance

**Best for**: Understanding trade-offs between adaptation strategies

---

### 4. Freezing Strategies
**Notebook**: `04_freezing_strategies.ipynb`

Learn how to:
- Freeze backbone layers
- Unfreeze top-K layers
- Always unfreeze layer norms
- Implement progressive unfreezing

**Best for**: Transfer learning and low-data regimes

---

### 5. Conditioning Patterns
**Notebook**: `05_conditioning_patterns.ipynb`

Learn how to:
- Implement FiLM conditioning
- Use cross-attention for multi-modal inputs
- Apply classifier-free guidance
- Condition on perturbations and cell types

**Best for**: Controlled generation and perturbation modeling

---

### 6. Mixture of Experts
**Notebook**: `06_mixture_of_experts.ipynb`

Learn how to:
- Implement MoE layers
- Route tokens to experts
- Balance expert utilization
- Scale models efficiently

**Best for**: Advanced architectures and scaling

---

### 7. End-to-End Gene Expression
**Notebook**: `07_end_to_end_gene_expression.ipynb`

Learn how to:
- Build complete latent diffusion pipeline
- Train with LoRA on gene expression data
- Evaluate generation quality
- Deploy for inference

**Best for**: Complete practical application

---

## Prerequisites

### Environment Setup

```bash
# Activate conda environment
mamba activate genailab

# Verify installation
python -c "import genailab.foundation; print('Foundation package ready!')"
```

### Hardware Requirements

- **Minimum**: M1 Mac 16GB (small models)
- **Recommended**: GPU with 24GB+ (medium/large models)
- **Optimal**: Multi-GPU setup (distributed training)

The tutorials automatically detect your hardware and adjust accordingly.

---

## Learning Path

### For Beginners
1. Start with **01_model_sizes_and_resources**
2. Learn **02_lora_basics**
3. Try **07_end_to_end_gene_expression**

### For Practitioners
1. Review **03_adapters_vs_lora**
2. Explore **05_conditioning_patterns**
3. Implement **07_end_to_end_gene_expression**

### For Researchers
1. Study **06_mixture_of_experts**
2. Experiment with **05_conditioning_patterns**
3. Customize **07_end_to_end_gene_expression**

---

## Quick Start

```python
# In any notebook
from genailab.foundation.configs import get_resource_profile, get_model_config
from genailab.foundation.tuning import apply_lora_to_model

# Auto-detect and configure
profile = get_resource_profile()
config = get_model_config(profile.recommended_model_size)

print(f"Detected: {profile.name}")
print(f"Model size: {profile.recommended_model_size}")
print(f"Estimated memory: {config.memory_estimate_gb()}GB")
```

---

## Data Requirements

Most tutorials use synthetic data for demonstration. For real applications:

- **Gene expression**: scRNA-seq data (AnnData format)
- **Perturbation**: Perturb-seq data (Norman et al., Replogle et al.)
- **Bulk RNA-seq**: TCGA, GTEx datasets

See `genailab.data` for data loading utilities.

---

## Related Resources

### Documentation
- [Foundation Models Overview](../../docs/foundation_models/leveraging_foundation_models_v2.md)
- [Transformer Data Shapes](../../docs/foundation_models/data_shape_v2.md)
- [Latent Diffusion Series](../../docs/latent_diffusion/)

### Code
- [Foundation Package](../../src/genailab/foundation/)
- [Model Configs](../../src/genailab/foundation/configs/)
- [Tuning Modules](../../src/genailab/foundation/tuning/)

### Examples
- [Production Scripts](../../examples/foundation_models/)

---

## Contributing

To add a new tutorial:

1. Create notebook in this directory
2. Follow naming convention: `XX_topic_name.ipynb`
3. Include clear learning objectives
4. Use small synthetic data for fast iteration
5. Add entry to this README

---

## Support

For issues or questions:
- Check [Foundation README](../../src/genailab/foundation/README.md)
- Review [documentation](../../docs/foundation_models/)
- Open an issue on GitHub
