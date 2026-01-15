# Generative AI for Perturbation Modeling: Beyond GEM-1

**Author's Note**: This document analyzes GEM-1's supervised learning approach and proposes how generative AI (diffusion models, VAEs, flow-based models) could enhance perturbation prediction, particularly for scPerturb datasets.

---

## Executive Summary

**Your analysis is correct**: GEM-1 is a **conditional predictive model**, not a generative model in the diffusion/VAE/flow sense. It learns $\mathbb{E}[x \mid \text{metadata}]$, not $p(x \mid \text{metadata})$.

**Your assessment is also correct**: GEM-1's approach is realistic and may work better than pure generative AI for many applications—especially when you need **deterministic, interpretable predictions** rather than stochastic samples.

**However**, for perturbation modeling (especially scPerturb), generative AI offers unique advantages that supervised learning cannot provide:

1. **Uncertainty quantification** - Multiple plausible cellular responses
2. **Counterfactual generation** - "What if" scenarios beyond training data
3. **Compositional perturbations** - Combining unseen perturbations
4. **Cell-level heterogeneity** - Capturing biological variability
5. **Out-of-distribution robustness** - Generalizing to novel perturbations

---

## Feedback on Your GEM-1 Analysis

### What You Got Right

1. **GEM-1 is not generative** - No evidence of stochastic sampling, latent variables, or diffusion processes
2. **It's compilation, not hallucination** - Learning a dense lookup table over experimental conditions
3. **The innovation is data harmonization** - Metadata curation is the real breakthrough
4. **Supervised learning is pragmatic** - For many applications, $\mathbb{E}[x]$ is sufficient

### Critical Insights You Identified

**"Predictive model first → generative wrapper later"** - This is the correct architecture.

GEM-1 has solved the hard problem: **learning the conditional mean structure of biology**. This is essential infrastructure. Generative models can build on top of this foundation.

**Why they avoided diffusion/VAEs** - Your four reasons are spot-on:
- Ambiguous notion of "sample"
- Unclear noise model
- Difficult validation of novelty
- Metadata dominates variance

These are real challenges, but they're not insurmountable—they're design constraints.

### Where Generative AI Adds Value (Not Replaces)

GEM-1's approach is excellent for:
- **Interpolation** within the training distribution
- **Point predictions** for experimental planning
- **Label imputation** (e.g., predicting missing sex labels)

Generative AI is essential for:
- **Extrapolation** to novel perturbation combinations
- **Uncertainty-aware predictions** for risk assessment
- **Diversity generation** for data augmentation
- **Causal intervention modeling** with counterfactuals

---

## The Perturbation Modeling Challenge

### Why scPerturb is Different from Bulk RNA-seq

scPerturb datasets have unique characteristics that make them ideal for generative modeling:

1. **Single-cell resolution** - Natural notion of "sample" (one cell)
2. **Controlled perturbations** - Clear causal interventions (CRISPR, compounds)
3. **Biological variability** - Cells respond heterogeneously to the same perturbation
4. **Compositional structure** - Perturbations can be combined (multi-gene knockouts)
5. **Counterfactual pairs** - Control vs perturbed cells from same experiment

### What GEM-1 Cannot Do (By Design)

For a given perturbation, GEM-1 predicts:

$$
\hat{x}_{\text{perturbed}} = f(\text{cell type}, \text{perturbation}, \text{dose}, \text{time})
$$

This gives you **one expression profile** per condition.

But biology is stochastic. The same perturbation in the same cell type produces:
- Different responses in different cells
- Bimodal or multimodal distributions
- Rare subpopulations with extreme responses
- Temporal dynamics with variable kinetics

GEM-1's point estimate cannot capture this **biological uncertainty**.

---

## Proposed Generative AI Approaches for Perturbation Modeling

### Architecture 1: Conditional Diffusion on scPerturb

**Core Idea**: Learn $p(x_{\text{perturbed}} \mid x_{\text{control}}, \text{perturbation})$

#### Model Design

```
Input:
  - x_control: Gene expression of control cell (or population mean)
  - perturbation: One-hot or embedding of perturbation identity
  - metadata: Cell type, dose, time

Output:
  - x_perturbed ~ p(x | x_control, perturbation, metadata)
```

#### Training Objective

Use **denoising score matching** with perturbation conditioning:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| s_\theta(x_t, t, c) - \nabla_{x_t} \log p(x_t \mid x_0) \|^2 \right]
$$

where $c = [\text{perturbation}, \text{metadata}, x_{\text{control}}]$

#### Key Advantages

1. **Generates diverse cellular responses** - Sample multiple times to get population distribution
2. **Interpolates between perturbations** - Smooth perturbation space
3. **Composes perturbations** - Combine embeddings for multi-gene knockouts
4. **Uncertainty quantification** - Variance in samples reflects biological variability

#### Implementation Strategy

```python
class PerturbationDiffusion(nn.Module):
    def __init__(self, n_genes=5000, perturbation_dim=256):
        self.gene_encoder = GeneExpressionEncoder(n_genes)
        self.perturbation_encoder = PerturbationEncoder(perturbation_dim)
        self.unet = ConditionalUNet(
            in_channels=n_genes,
            condition_dim=perturbation_dim + metadata_dim + n_genes
        )
        self.sde = VPSDE(schedule='cosine')
    
    def forward(self, x_t, t, x_control, perturbation, metadata):
        # Encode conditioning
        pert_emb = self.perturbation_encoder(perturbation)
        control_emb = self.gene_encoder(x_control)
        condition = torch.cat([pert_emb, metadata, control_emb], dim=-1)
        
        # Predict score
        score = self.unet(x_t, t, condition)
        return score
```

---

### Architecture 2: Causal VAE with Perturbation Operators

**Core Idea**: Learn disentangled latent space where perturbations are **linear operators**

#### Model Design

Inspired by **causal representation learning**, decompose latent space into:

$$
z = [z_{\text{cell identity}}, z_{\text{cell state}}, z_{\text{technical}}]
$$

Perturbations act as transformations:

$$
z_{\text{perturbed}} = z_{\text{control}} + \Delta_{\text{perturbation}}
$$

#### Training Objective

Combine VAE ELBO with **causal regularization**:

$$
\mathcal{L} = \mathcal{L}_{\text{ELBO}} + \lambda_1 \mathcal{L}_{\text{disentangle}} + \lambda_2 \mathcal{L}_{\text{causal}}
$$

where:

- $\mathcal{L}_{\text{disentangle}}$ encourages independence of latent factors
- $\mathcal{L}_{\text{causal}}$ enforces perturbation effects are additive in latent space

#### Key Advantages

1. **Interpretable perturbation effects** - $\Delta$ vectors represent causal interventions
2. **Compositional generalization** - $\Delta_1 + \Delta_2$ for combined perturbations
3. **Transfer across cell types** - Learn universal perturbation operators
4. **Counterfactual generation** - Apply perturbation to any cell

#### Validation Strategy

Test on held-out perturbations:
- **Interpolation**: Unseen doses of known perturbations
- **Extrapolation**: Unseen combinations of known perturbations
- **Transfer**: Known perturbations in unseen cell types

---

### Architecture 3: Flow-Based Perturbation Model (Optimal Transport)

**Core Idea**: Learn the **transport map** from control distribution to perturbed distribution

#### Model Design

Use **continuous normalizing flows** (CNF) to model:

$$
\frac{d x}{d t} = f_\theta(x, t, \text{perturbation})
$$

This learns the **trajectory** of cellular response over time.

#### Training Objective

Minimize **optimal transport cost** between control and perturbed distributions:

$$
\mathcal{L} = \mathbb{E}_{x_{\text{control}}, x_{\text{perturbed}}} \left[ \| x_{\text{perturbed}} - \Phi_\theta(x_{\text{control}}, \text{perturbation}) \|^2 \right]
$$

where $\Phi_\theta$ is the learned flow.

#### Key Advantages

1. **Exact likelihood** - No variational approximation
2. **Invertible** - Can go from perturbed → control
3. **Temporal dynamics** - Natural interpretation as time evolution
4. **Efficient sampling** - Single forward pass (no iterative denoising)

---

## Hybrid Architecture: GEM-1 + Generative Wrapper

### The Best of Both Worlds

**Stage 1: GEM-1-style Predictive Model**

- Learn $\mu(c) = \mathbb{E}[x \mid c]$ for all conditions $c$
- Massive scale, data harmonization, metadata curation
- Provides strong **prior** for generative model

**Stage 2: Generative Model on Residuals**

- Learn $p(x - \mu(c) \mid c)$ - the distribution around the mean
- Captures biological variability, technical noise, rare events
- Much easier to learn than full $p(x \mid c)$

### Implementation

```python
class HybridPerturbationModel:
    def __init__(self):
        # Stage 1: Predictive (GEM-1 style)
        self.mean_predictor = ConditionalMeanModel()
        
        # Stage 2: Generative (diffusion on residuals)
        self.residual_diffusion = ResidualDiffusion()
    
    def predict(self, condition):
        # Deterministic mean
        mu = self.mean_predictor(condition)
        return mu
    
    def sample(self, condition, n_samples=100):
        # Mean + stochastic residuals
        mu = self.mean_predictor(condition)
        residuals = self.residual_diffusion.sample(condition, n_samples)
        return mu + residuals
```

### Why This Works

1. **Mean prediction is stable** - Supervised learning excels here
2. **Residual distribution is simpler** - Centered at zero, easier to model
3. **Separates signal from noise** - Biological vs technical variability
4. **Leverages both paradigms** - Predictive accuracy + generative flexibility

---

## Concrete Proposal for genai-lab

### Phase 1: Proof of Concept (scPerturb Subset)

**Dataset**: Norman et al. 2019 (Perturb-seq, K562 cells)
- ~250,000 cells
- ~5,000 genes
- ~100 perturbations (single-gene CRISPR knockouts)
- Control cells available

**Model**: Conditional diffusion (Architecture 1)

**Metrics**:

- **Reconstruction**: MSE on held-out cells
- **Diversity**: Variance of generated samples vs real
- **Composition**: Accuracy on held-out double knockouts
- **Biological validity**: Pathway enrichment, known gene interactions

**Timeline**: 2-3 weeks

### Phase 2: Scaling (Full scPerturb)

**Dataset**: All scPerturb datasets
- Multiple cell types
- Multiple perturbation modalities (CRISPR, compounds, overexpression)
- Varying doses and timepoints

**Model**: Causal VAE (Architecture 2) or Hybrid (GEM-1 + diffusion)

**Metrics**:

- **Transfer learning**: Train on cell line, test on primary cells
- **Zero-shot perturbations**: Predict unseen perturbations from embeddings
- **Counterfactuals**: Generate "what if" scenarios

**Timeline**: 1-2 months

### Phase 3: Integration with GEM-1 Philosophy

**Data Harmonization**:

- Apply GEM-1's metadata curation to scPerturb
- Standardize perturbation ontologies
- Align cell type annotations

**Model Architecture**:

- Use GEM-1-style predictive model as initialization
- Add generative wrapper for uncertainty
- Multi-task learning: predict mean + sample distribution

**Validation**:

- Compare against GEM-1 predictions (if available)
- Benchmark on experimental validation datasets
- Collaborate with experimentalists for prospective validation

---

## Key Differences from GEM-1

| Aspect | GEM-1 | Generative AI (Proposed) |
|--------|-------|--------------------------|
| **Output** | Single prediction | Distribution of outcomes |
| **Uncertainty** | None (point estimate) | Explicit (sample variance) |
| **Novelty** | Interpolation only | Extrapolation possible |
| **Composition** | Limited | Natural (latent arithmetic) |
| **Validation** | Prediction accuracy | Diversity + accuracy |
| **Use case** | Experimental planning | Data augmentation, counterfactuals |

---

## Why Generative AI is Complementary, Not Competitive

GEM-1 and generative models solve **different problems**:

**GEM-1 answers**: "What is the expected expression profile for this condition?"
- Essential for: experimental design, hypothesis generation, label imputation

**Generative AI answers**: "What are all the possible expression profiles for this condition?"
- Essential for: risk assessment, rare event prediction, synthetic data generation

**Both are needed** for a complete perturbation modeling system.

---

## Technical Challenges and Solutions

### Challenge 1: High Dimensionality (5,000-20,000 genes)

**Solution**: 

- Use **gene program embeddings** (PCA, NMF, or learned)
- Model in low-dimensional latent space (~50-200 dims)
- Decode back to gene space

### Challenge 2: Sparse, Zero-Inflated Data

**Solution**:

- Use **zero-inflated loss functions**
- Separate models for dropout vs expression level
- Or use **scVI-style** probabilistic framework

### Challenge 3: Batch Effects

**Solution**:

- Include batch as conditioning variable
- Use **adversarial training** to remove batch effects
- Or **CycleGAN-style** batch correction in latent space

### Challenge 4: Limited Perturbation Coverage

**Solution**:

- **Meta-learning** across perturbations
- **Transfer learning** from related perturbations
- **Graph neural networks** over perturbation similarity graphs

### Challenge 5: Validation Without Ground Truth

**Solution**:

- **Biological consistency checks**: pathway enrichment, known interactions
- **Cross-validation**: held-out perturbations, cell types, doses
- **Prospective validation**: generate predictions → experimentalists test

---

## Recommended Reading

### Perturbation Modeling
- **scGen** (Lotfollahi et al. 2019) - VAE for perturbation prediction
- **CPA** (Lotfollahi et al. 2023) - Compositional perturbation autoencoder
- **GEARS** (Roohani et al. 2023) - Graph neural network for genetic perturbations

### Causal Representation Learning
- **CATE** (Schwab et al. 2020) - Causal effect VAE
- **Causal-BALD** (Jesson et al. 2021) - Bayesian active learning for causal discovery

### Diffusion for Biology
- **scDiffusion** (Yang et al. 2023) - Diffusion models for single-cell data
- **DiffCSP** (Jing et al. 2023) - Diffusion for crystal structure prediction (similar principles)

---

## Next Steps for genai-lab

1. **Implement baseline** - Conditional diffusion on Norman et al. dataset
2. **Benchmark against scGen/CPA** - Compare generative quality
3. **Ablation studies** - Conditioning strategies, architecture choices
4. **Biological validation** - Pathway analysis, known gene interactions
5. **Scale to full scPerturb** - Multi-dataset, multi-modality
6. **Hybrid model** - Integrate GEM-1-style predictive component

---

## Conclusion

**Your analysis is sharp and correct**: GEM-1 is not a generative model, and its supervised learning approach is pragmatic and effective for many applications.

**However**, for perturbation modeling—especially with scPerturb—generative AI offers unique capabilities:
- Uncertainty quantification
- Compositional generalization
- Counterfactual reasoning
- Biological variability modeling

**The optimal path forward** is not generative AI *instead of* GEM-1, but generative AI *on top of* GEM-1's data harmonization and predictive foundation.

**genai-lab is well-positioned** to explore this hybrid approach, combining:
- GEM-1's data curation philosophy
- Diffusion models' generative flexibility
- scPerturb's causal perturbation structure

This could lead to a **next-generation perturbation modeling system** that provides both accurate predictions and biologically meaningful uncertainty.
