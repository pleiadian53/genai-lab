# Foundation Models for Computational Biology

**Adapting large-scale foundation models for gene expression and multi-omics tasks.**

---

## Overview

Foundation models trained on massive biological datasets (DNA, RNA, protein) are emerging as powerful tools for computational biology. This section covers practical strategies for **adapting** these models to specific tasks without training from scratch.

**Key Topics:**

- 🎯 **Model Selection**: Choosing the right foundation model for your task
- 🔧 **Adaptation Strategies**: LoRA, adapters, fine-tuning, freezing
- 📊 **Data Preparation**: Handling gene expression, sequences, and multi-omics
- 💻 **Implementation**: Resource-aware configs, hardware optimization
- 🚀 **Deployment**: Inference, serving, and production pipelines

---

## Key Documents

### [Leveraging Foundation Models](leveraging_foundation_models_v2.md)

**Comprehensive guide to foundation model adaptation:**

- Overview of available models (Geneformer, scGPT, BigRNA, ESM, etc.)
- When to use foundation models vs. train from scratch
- Adaptation strategies (LoRA, adapters, full fine-tuning)
- Conditioning and control (FiLM, cross-attention, CFG)
- Resource management (small/medium/large configs)

**Best for:** Understanding the landscape and choosing an adaptation strategy

---

### [Data Shape & Tensors](data_shape_v2.md)

**How to prepare your data for foundation models:**

- Input representations (tokens, embeddings, sequences)
- Batch shapes and padding strategies
- Attention masks and position encodings
- Cell type, drug, and perturbation conditioning
- Multi-omics integration

**Best for:** Implementing data loaders and preprocessing pipelines

---

### [Implementation Guide](IMPLEMENTATION_GUIDE.md)

**Step-by-step implementation for common tasks:**

- Setting up environments and dependencies
- Loading pre-trained models
- Implementing LoRA and adapters
- Training loops with mixed precision
- Evaluation and benchmarking

**Best for:** Hands-on implementation and code examples

---

## Why Foundation Models for Biology?

### Traditional ML Approach
```
Custom model → Train from scratch → High data requirements → Task-specific
```

### Foundation Model Approach
```
Pre-trained model → Adapt (LoRA/fine-tune) → Low data requirements → Transferable
```

**Advantages:**

✅ **Sample efficiency**: Learn from 100s-1000s of examples vs. millions  
✅ **Transfer learning**: Leverage knowledge from massive pre-training datasets  
✅ **Generalization**: Better performance on out-of-distribution data  
✅ **Multi-task**: Single model for multiple downstream tasks  
✅ **Interpretability**: Pre-learned biological representations

---

## Available Foundation Models (2026)

### Gene Expression & Multi-Omics

| Model | Organization | Focus | Size | Open Source |
|-------|-------------|-------|------|-------------|
| **GEM-1** | Synthesize Bio | Gene expression generation | Unknown | ❌ |
| **BigRNA** | Deep Genomics | RNA biology | ~2B params | ❌ |
| **Geneformer** | Theodoris et al. | Single-cell transfer learning | 10M-100M | ✅ |
| **scGPT** | Cui et al. | Single-cell foundation | 10M-100M | ✅ |

### DNA & RNA Sequences

| Model | Organization | Focus | Size | Open Source |
|-------|-------------|-------|------|-------------|
| **Evo 2** | Arc Institute | DNA sequence (8kb context) | 7B params | ✅ |
| **Nucleotide Transformer** | InstaDeep | Multi-species DNA | 500M-2.5B | ✅ |
| **Helix-mRNA** | Helical | mRNA sequences | Unknown | ✅ |

### Protein & Structure

| Model | Organization | Focus | Size | Open Source |
|-------|-------------|-------|------|-------------|
| **ESM3** | EvolutionaryScale | Protein design | 1.4B-98B | ✅ (7B, 98B) |
| **AlphaFold 3** | Isomorphic Labs | Protein structure | Unknown | ❌ |
| **Chai-1** | Chai Discovery | Antibody design | Unknown | ✅ |

---

## Typical Workflow

### 1. **Choose Your Model**
Select based on:
- Input type (expression, sequence, structure)
- Task (generation, prediction, classification)
- Available compute (model size)
- Open source vs. proprietary

### 2. **Prepare Your Data**
- Tokenize or embed inputs
- Create attention masks
- Add condition labels (cell type, drug, etc.)
- Split train/val/test

### 3. **Select Adaptation Strategy**

| Strategy | Data Needed | Compute | Best For |
|----------|-------------|---------|----------|
| **Frozen + Linear Probe** | 100s | Low | Quick prototyping |
| **LoRA** | 1000s | Medium | Most tasks (recommended) |
| **Adapter Layers** | 1000s | Medium | Multi-task learning |
| **Full Fine-Tuning** | 10,000s+ | High | Maximum performance |

### 4. **Train & Evaluate**
- Use mixed precision (fp16/bfloat16)
- Monitor overfitting (small datasets)
- Validate on held-out cell types/drugs
- Compare to baselines

### 5. **Deploy**
- Quantize for inference (int8, int4)
- Batch predictions for efficiency
- Monitor uncertainty estimates

---

## Example Use Cases

### 🧬 Drug Response Prediction
**Input:** Baseline gene expression + drug ID  
**Output:** Perturbed gene expression  
**Model:** Fine-tuned Geneformer with LoRA  
**Data:** Perturb-seq, LINCS L1000

### 🔬 Cell Type Annotation
**Input:** Single-cell expression profile  
**Output:** Cell type label  
**Model:** Frozen scGPT + classifier head  
**Data:** Tabula Sapiens, CellxGene

### 💊 Combination Therapy
**Input:** Expression + drug A + drug B  
**Output:** Synergy score  
**Model:** Multi-task LoRA adapter  
**Data:** DrugComb, O'Neil et al.

### 🧪 RNA Design
**Input:** Target structure + constraints  
**Output:** RNA sequence  
**Model:** Fine-tuned Helix-mRNA  
**Data:** RNAcentral, Rfam

---

## Getting Started

**Recommended learning path:**

1. Start with [Leveraging Foundation Models](leveraging_foundation_models_v2.md) for conceptual overview
2. Follow [Data Shape & Tensors](data_shape_v2.md) to prepare your data
3. Use [Implementation Guide](IMPLEMENTATION_GUIDE.md) for hands-on coding
4. Experiment with different adaptation strategies (LoRA → Adapters → Full fine-tune)
5. Evaluate on held-out data and compare to baselines

**Next steps:**

- 📓 **Notebooks**: Coming soon - interactive tutorials for each model
- 🔧 **Code examples**: See `examples/foundation_models/` for production scripts
- 📚 **Theory**: Explore [DiT](../DiT/), [JEPA](../JEPA/), [Latent Diffusion](../latent_diffusion/) for advanced architectures

---

## References

### Key Papers

- **Geneformer**: Theodoris et al. (2023) - "Transfer learning enables predictions in network biology"
- **scGPT**: Cui et al. (2024) - "scGPT: Toward building a foundation model for single-cell multi-omics"
- **ESM3**: Hayes et al. (2024) - "Simulating 500 million years of evolution"
- **Nucleotide Transformer**: Dalla-Torre et al. (2023) - "The Nucleotide Transformer"

### Industry Reports

- [Foundation Models for Computational Biology](https://www.nature.com/articles/s41592-023-01905-2) - Nature Methods
- [17 Companies Pioneering AI Foundation Models in Pharma](https://labiotech.eu/lists/ai-foundation-models-drug-discovery/)
- [NVIDIA BioNeMo Platform](https://www.nvidia.com/en-us/clara/bionemo/)

---

**Questions or suggestions?** Open an issue on [GitHub](https://github.com/pleiadian53/genai-lab/issues)
