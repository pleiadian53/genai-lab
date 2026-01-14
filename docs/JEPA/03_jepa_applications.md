# JEPA Applications: From Vision to Biology

This document maps JEPA concepts from computer vision to computational biology, covering perturbation prediction, trajectory inference, multi-omics integration, and drug response modeling.

**Prerequisites**: Understanding of [JEPA overview](00_jepa_overview.md), [foundations](01_jepa_foundations.md), and [training](02_jepa_training.md).

---

## Overview

### Vision → Biology Mapping

| Vision Domain | Biology Domain | JEPA Task |
|---------------|----------------|-----------|
| **Image masking** | Gene expression masking | Predict masked genes from visible |
| **Video prediction** | Time-series prediction | Predict future states from past |
| **Frame interpolation** | Trajectory interpolation | Fill gaps in developmental paths |
| **Action conditioning** | Perturbation conditioning | Predict perturbed state from baseline |
| **Multi-view learning** | Multi-omics integration | Predict one modality from another |

**Key insight**: Replace "pixels" with "genes", "frames" with "time points", "actions" with "perturbations".

---

## 1. Perturbation Prediction (Perturb-seq)

### 1.1 Problem Setup

**Goal**: Predict cellular response to genetic/chemical perturbations

**Data**:
- Baseline expression: $x_0 \in \mathbb{R}^{20000}$ (unperturbed cells)
- Perturbed expression: $x_p \in \mathbb{R}^{20000}$ (after perturbation)
- Perturbation info: Gene ID, type (KO/OE), dose

**JEPA formulation**:
```
Context: Baseline expression + perturbation info
Target: Perturbed expression
Task: Predict z_perturbed from z_baseline and perturbation
```

### 1.2 Architecture

```python
class PerturbationJEPA(nn.Module):
    """
    JEPA for perturbation prediction.
    
    Args:
        num_genes: Number of genes
        embed_dim: Embedding dimension
        num_tokens: Number of tokens
        perturbation_dim: Perturbation embedding dimension
    """
    def __init__(
        self,
        num_genes=20000,
        embed_dim=256,
        num_tokens=64,
        perturbation_dim=128,
    ):
        super().__init__()
        
        # Encoder for gene expression
        self.encoder = GeneExpressionEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
        )
        
        # Perturbation encoder
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(num_genes + 10, perturbation_dim),  # gene_id + metadata
            nn.LayerNorm(perturbation_dim),
            nn.GELU(),
            nn.Linear(perturbation_dim, perturbation_dim),
        )
        
        # Conditional predictor
        self.predictor = ConditionalPredictor(
            embed_dim=embed_dim,
            condition_dim=perturbation_dim,
            depth=4,
            num_heads=8,
        )
        
        # VICReg loss
        self.vicreg = VICRegLoss()
    
    def forward(self, x_baseline, x_perturbed, perturbation_info):
        """
        Args:
            x_baseline: Baseline expression (B, num_genes)
            x_perturbed: Perturbed expression (B, num_genes)
            perturbation_info: Perturbation metadata (B, num_genes + 10)
        
        Returns:
            loss: Total loss
            loss_dict: Loss components
        """
        # Encode baseline and perturbed
        z_baseline = self.encoder(x_baseline)
        z_perturbed = self.encoder(x_perturbed)
        
        # Encode perturbation
        pert_emb = self.perturbation_encoder(perturbation_info)
        
        # Predict perturbed from baseline + perturbation
        z_pred = self.predictor(z_baseline, pert_emb)
        
        # VICReg loss
        loss, loss_dict = self.vicreg(z_pred, z_perturbed)
        
        return loss, loss_dict
    
    @torch.no_grad()
    def predict_perturbation(self, x_baseline, perturbation_info):
        """
        Predict perturbed state.
        
        Args:
            x_baseline: Baseline expression (B, num_genes)
            perturbation_info: Perturbation metadata (B, num_genes + 10)
        
        Returns:
            z_pred: Predicted perturbed embedding (B, num_tokens, embed_dim)
        """
        z_baseline = self.encoder(x_baseline)
        pert_emb = self.perturbation_encoder(perturbation_info)
        z_pred = self.predictor(z_baseline, pert_emb)
        return z_pred
```

### 1.3 Training

```python
# Dataset
from datasets import load_perturbseq_data

baseline_expr, perturbed_expr, pert_info = load_perturbseq_data('norman2019')

dataset = PerturbSeqDataset(
    baseline_expr=baseline_expr,
    perturbed_expr=perturbed_expr,
    perturbation_info=pert_info,
)

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model
model = PerturbationJEPA(
    num_genes=20000,
    embed_dim=256,
    num_tokens=64,
)

# Train
train_jepa_complete(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-3,
)
```

### 1.4 Evaluation

**Metrics**:
1. **Embedding similarity**: Cosine similarity between predicted and actual embeddings
2. **DEG recovery**: Fraction of differentially expressed genes correctly predicted
3. **Pathway consistency**: Predicted perturbations affect correct pathways
4. **Held-out perturbations**: Generalization to unseen perturbations

```python
@torch.no_grad()
def evaluate_perturbation_prediction(model, test_loader, device):
    """
    Evaluate perturbation prediction.
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_similarities = []
    all_deg_recalls = []
    
    for x_baseline, x_perturbed, pert_info in test_loader:
        x_baseline = x_baseline.to(device)
        x_perturbed = x_perturbed.to(device)
        pert_info = pert_info.to(device)
        
        # Predict
        z_pred = model.predict_perturbation(x_baseline, pert_info)
        
        # Actual
        z_actual = model.encoder(x_perturbed)
        
        # Cosine similarity
        z_pred_flat = z_pred.mean(dim=1)  # Average over tokens
        z_actual_flat = z_actual.mean(dim=1)
        similarity = F.cosine_similarity(z_pred_flat, z_actual_flat, dim=1)
        all_similarities.append(similarity.cpu())
        
        # DEG recovery (if we have a decoder)
        # ...
    
    metrics = {
        'embedding_similarity': torch.cat(all_similarities).mean().item(),
        # 'deg_recall': ...,
        # 'pathway_consistency': ...,
    }
    
    return metrics
```

### 1.5 Advantages Over Existing Methods

**Comparison with scGen/CPA**:

| Aspect | scGen/CPA | JEPA |
|--------|-----------|------|
| **Architecture** | VAE + arithmetic | Encoder + Predictor |
| **Perturbation** | Latent arithmetic | Learned operators |
| **Reconstruction** | Required | Not needed |
| **Efficiency** | Moderate | High (no decoder) |
| **Compositional** | Limited | Natural |
| **Generalization** | Moderate | Better (learned operators) |

**JEPA advantages**:
1. **No reconstruction** — Focus on prediction, not generation
2. **Learned operators** — Perturbations are learned, not hand-crafted
3. **Compositional** — Naturally combine multiple perturbations
4. **Efficient** — No decoder, faster training

---

## 2. Trajectory Inference

### 2.1 Problem Setup

**Goal**: Predict developmental or disease trajectories

**Data**:
- Time-series expression: $\{x_{t_1}, x_{t_2}, ..., x_{t_n}\}$
- Time points: $\{t_1, t_2, ..., t_n\}$

**JEPA formulation**:
```
Context: Expression at time t
Target: Expression at time t+Δt
Task: Predict z_{t+Δt} from z_t and Δt
```

### 2.2 Architecture

```python
class TrajectoryJEPA(nn.Module):
    """
    JEPA for trajectory inference.
    
    Predicts future cell states from current state and time.
    """
    def __init__(
        self,
        num_genes=20000,
        embed_dim=256,
        num_tokens=64,
        time_embed_dim=64,
    ):
        super().__init__()
        
        # Encoder
        self.encoder = GeneExpressionEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
        )
        
        # Time encoder
        self.time_encoder = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
        )
        
        # Predictor
        self.predictor = ConditionalPredictor(
            embed_dim=embed_dim,
            condition_dim=time_embed_dim,
            depth=4,
        )
        
        self.vicreg = VICRegLoss()
    
    def forward(self, x_t, x_t_next, delta_t):
        """
        Args:
            x_t: Expression at time t (B, num_genes)
            x_t_next: Expression at time t+Δt (B, num_genes)
            delta_t: Time difference (B,)
        
        Returns:
            loss: Total loss
            loss_dict: Loss components
        """
        # Encode current and future
        z_t = self.encoder(x_t)
        z_t_next = self.encoder(x_t_next)
        
        # Encode time
        time_emb = self.time_encoder(delta_t)
        
        # Predict future from current + time
        z_pred = self.predictor(z_t, time_emb)
        
        # VICReg loss
        loss, loss_dict = self.vicreg(z_pred, z_t_next)
        
        return loss, loss_dict
    
    @torch.no_grad()
    def predict_trajectory(self, x_start, time_points):
        """
        Predict trajectory from starting point.
        
        Args:
            x_start: Starting expression (B, num_genes)
            time_points: List of future time points
        
        Returns:
            trajectory: Predicted embeddings at each time point
        """
        trajectory = []
        z_current = self.encoder(x_start)
        
        for t in time_points:
            time_emb = self.time_encoder(torch.tensor([t]))
            z_next = self.predictor(z_current, time_emb)
            trajectory.append(z_next)
            z_current = z_next
        
        return torch.stack(trajectory, dim=1)  # (B, num_timepoints, num_tokens, embed_dim)
```

### 2.3 Applications

**1. Developmental biology**:
- Predict cell differentiation trajectories
- Identify branch points
- Infer lineage relationships

**2. Disease progression**:
- Predict disease state evolution
- Identify critical transitions
- Stratify patients by trajectory

**3. Drug response over time**:
- Predict temporal response to drugs
- Identify optimal treatment timing
- Detect resistance emergence

---

## 3. Multi-Omics Integration

### 3.1 Problem Setup

**Goal**: Predict one modality from another

**Data**:
- RNA-seq: $x_{rna} \in \mathbb{R}^{20000}$
- Protein: $x_{protein} \in \mathbb{R}^{5000}$
- ATAC-seq: $x_{atac} \in \mathbb{R}^{50000}$

**JEPA formulation**:
```
Context: RNA-seq
Target: Protein expression
Task: Predict z_protein from z_rna
```

### 3.2 Architecture

```python
class MultiOmicsJEPA(nn.Module):
    """
    JEPA for multi-omics integration.
    
    Predicts one omics modality from another.
    """
    def __init__(
        self,
        rna_dim=20000,
        protein_dim=5000,
        embed_dim=256,
        num_tokens=64,
    ):
        super().__init__()
        
        # Modality-specific encoders
        self.rna_encoder = GeneExpressionEncoder(
            num_genes=rna_dim,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
        )
        
        self.protein_encoder = GeneExpressionEncoder(
            num_genes=protein_dim,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
        )
        
        # Cross-modality predictor
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            depth=6,
        )
        
        self.vicreg = VICRegLoss()
    
    def forward(self, x_rna, x_protein):
        """
        Args:
            x_rna: RNA-seq (B, rna_dim)
            x_protein: Protein (B, protein_dim)
        
        Returns:
            loss: Total loss
            loss_dict: Loss components
        """
        # Encode both modalities
        z_rna = self.rna_encoder(x_rna)
        z_protein = self.protein_encoder(x_protein)
        
        # Predict protein from RNA
        z_protein_pred = self.predictor(z_rna)
        
        # VICReg loss
        loss, loss_dict = self.vicreg(z_protein_pred, z_protein)
        
        return loss, loss_dict
    
    @torch.no_grad()
    def predict_protein_from_rna(self, x_rna):
        """
        Predict protein expression from RNA-seq.
        
        Args:
            x_rna: RNA-seq (B, rna_dim)
        
        Returns:
            z_protein_pred: Predicted protein embedding
        """
        z_rna = self.rna_encoder(x_rna)
        z_protein_pred = self.predictor(z_rna)
        return z_protein_pred
```

### 3.3 Applications

**1. RNA → Protein prediction**:
- Predict protein abundance from transcriptomics
- Identify post-transcriptional regulation
- Validate proteomics experiments

**2. ATAC → RNA prediction**:
- Predict gene expression from chromatin accessibility
- Identify regulatory relationships
- Infer transcription factor activity

**3. Cross-species translation**:
- Predict human expression from mouse
- Transfer knowledge across species
- Validate evolutionary conservation

---

## 4. Drug Response Prediction

### 4.1 Problem Setup

**Goal**: Predict cellular response to drugs

**Data**:
- Baseline expression: $x_0$
- Drug features: Chemical structure, target, dose
- Treated expression: $x_{drug}$

**JEPA formulation**:
```
Context: Baseline + drug features
Target: Treated expression
Task: Predict z_treated from z_baseline and drug
```

### 4.2 Architecture

```python
class DrugResponseJEPA(nn.Module):
    """
    JEPA for drug response prediction.
    
    Predicts cellular response to drug treatment.
    """
    def __init__(
        self,
        num_genes=20000,
        drug_feature_dim=512,  # e.g., Morgan fingerprints
        embed_dim=256,
        num_tokens=64,
    ):
        super().__init__()
        
        # Expression encoder
        self.expr_encoder = GeneExpressionEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
        )
        
        # Drug encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Predictor
        self.predictor = ConditionalPredictor(
            embed_dim=embed_dim,
            condition_dim=embed_dim,
            depth=6,
        )
        
        self.vicreg = VICRegLoss()
    
    def forward(self, x_baseline, x_treated, drug_features):
        """
        Args:
            x_baseline: Baseline expression (B, num_genes)
            x_treated: Treated expression (B, num_genes)
            drug_features: Drug features (B, drug_feature_dim)
        
        Returns:
            loss: Total loss
            loss_dict: Loss components
        """
        # Encode baseline and treated
        z_baseline = self.expr_encoder(x_baseline)
        z_treated = self.expr_encoder(x_treated)
        
        # Encode drug
        drug_emb = self.drug_encoder(drug_features)
        
        # Predict treated from baseline + drug
        z_pred = self.predictor(z_baseline, drug_emb)
        
        # VICReg loss
        loss, loss_dict = self.vicreg(z_pred, z_treated)
        
        return loss, loss_dict
    
    @torch.no_grad()
    def predict_drug_response(self, x_baseline, drug_features):
        """
        Predict response to drug.
        
        Args:
            x_baseline: Baseline expression (B, num_genes)
            drug_features: Drug features (B, drug_feature_dim)
        
        Returns:
            z_pred: Predicted response embedding
        """
        z_baseline = self.expr_encoder(x_baseline)
        drug_emb = self.drug_encoder(drug_features)
        z_pred = self.predictor(z_baseline, drug_emb)
        return z_pred
    
    @torch.no_grad()
    def screen_drugs(self, x_baseline, drug_library):
        """
        Screen library of drugs.
        
        Args:
            x_baseline: Baseline expression (1, num_genes)
            drug_library: Library of drug features (N, drug_feature_dim)
        
        Returns:
            responses: Predicted responses for each drug (N, num_tokens, embed_dim)
        """
        # Encode baseline once
        z_baseline = self.expr_encoder(x_baseline)
        z_baseline = z_baseline.repeat(len(drug_library), 1, 1)
        
        # Encode all drugs
        drug_embs = self.drug_encoder(drug_library)
        
        # Predict responses
        responses = self.predictor(z_baseline, drug_embs)
        
        return responses
```

### 4.3 Applications

**1. Drug screening**:
- Predict response to large drug libraries
- Identify promising candidates
- Prioritize experiments

**2. Combination therapy**:
- Predict response to drug combinations
- Identify synergistic pairs
- Optimize dosing

**3. Patient stratification**:
- Predict patient-specific responses
- Personalize treatment
- Identify biomarkers

---

## 5. Combining JEPA with Generative Models

### 5.1 JEPA + Diffusion

**Motivation**: JEPA predicts embeddings, diffusion generates samples

```python
class JEPADiffusionHybrid(nn.Module):
    """
    Hybrid JEPA + Diffusion model.
    
    JEPA predicts perturbed embedding.
    Diffusion generates samples from embedding.
    """
    def __init__(
        self,
        jepa_model,
        diffusion_decoder,
    ):
        super().__init__()
        
        self.jepa = jepa_model
        self.diffusion = diffusion_decoder
    
    @torch.no_grad()
    def predict_and_generate(
        self,
        x_baseline,
        perturbation_info,
        num_samples=100,
    ):
        """
        Predict perturbed state and generate samples.
        
        Args:
            x_baseline: Baseline expression (B, num_genes)
            perturbation_info: Perturbation metadata
            num_samples: Number of samples to generate
        
        Returns:
            samples: Generated perturbed samples (B, num_samples, num_genes)
            z_pred: Predicted embedding (B, num_tokens, embed_dim)
        """
        # JEPA: Predict perturbed embedding
        z_pred = self.jepa.predict_perturbation(x_baseline, perturbation_info)
        
        # Diffusion: Generate samples from embedding
        samples = []
        for _ in range(num_samples):
            sample = self.diffusion.sample(z_pred)
            samples.append(sample)
        
        samples = torch.stack(samples, dim=1)  # (B, num_samples, num_genes)
        
        return samples, z_pred
```

**Benefits**:
1. **Prediction** — JEPA provides point estimate
2. **Uncertainty** — Diffusion provides distribution
3. **Efficiency** — JEPA is fast, diffusion only for final generation
4. **Best of both** — Combine prediction and generation

### 5.2 Training Strategy

**Two-stage training**:

**Stage 1: Train JEPA**
```python
# Train JEPA on prediction task
train_jepa(jepa_model, train_loader, num_epochs=100)
```

**Stage 2: Train Diffusion Decoder**
```python
# Freeze JEPA encoder
for param in jepa_model.encoder.parameters():
    param.requires_grad = False

# Train diffusion to decode embeddings
train_diffusion_decoder(
    diffusion_decoder,
    jepa_model.encoder,
    train_loader,
    num_epochs=50,
)
```

**Joint fine-tuning** (optional):
```python
# Fine-tune both together
for param in jepa_model.parameters():
    param.requires_grad = True

train_hybrid(hybrid_model, train_loader, num_epochs=20)
```

---

## 6. Evaluation Strategies

### 6.1 Embedding-Level Metrics

**1. Cosine similarity**:
```python
similarity = F.cosine_similarity(z_pred, z_actual, dim=-1).mean()
```

**2. L2 distance**:
```python
distance = torch.norm(z_pred - z_actual, dim=-1).mean()
```

**3. Rank correlation**:
```python
from scipy.stats import spearmanr

# Flatten embeddings
z_pred_flat = z_pred.flatten()
z_actual_flat = z_actual.flatten()

correlation, p_value = spearmanr(z_pred_flat, z_actual_flat)
```

### 6.2 Biological Metrics

**1. DEG recovery**:
```python
def compute_deg_recovery(pred_expr, actual_expr, baseline_expr, threshold=1.5):
    """
    Compute fraction of DEGs correctly predicted.
    
    Args:
        pred_expr: Predicted expression
        actual_expr: Actual expression
        baseline_expr: Baseline expression
        threshold: Fold-change threshold for DEG
    
    Returns:
        recall: Fraction of actual DEGs predicted
        precision: Fraction of predicted DEGs correct
    """
    # Actual DEGs
    actual_fc = actual_expr / (baseline_expr + 1e-6)
    actual_degs = (actual_fc > threshold) | (actual_fc < 1/threshold)
    
    # Predicted DEGs
    pred_fc = pred_expr / (baseline_expr + 1e-6)
    pred_degs = (pred_fc > threshold) | (pred_fc < 1/threshold)
    
    # Compute metrics
    true_positives = (actual_degs & pred_degs).sum()
    recall = true_positives / actual_degs.sum()
    precision = true_positives / pred_degs.sum()
    
    return recall.item(), precision.item()
```

**2. Pathway enrichment**:
```python
from gseapy import enrichr

def compute_pathway_consistency(pred_expr, actual_expr, baseline_expr):
    """
    Check if predicted DEGs enrich for same pathways as actual DEGs.
    """
    # Get actual DEGs
    actual_fc = actual_expr / (baseline_expr + 1e-6)
    actual_deg_genes = get_top_genes(actual_fc, top_k=200)
    
    # Get predicted DEGs
    pred_fc = pred_expr / (baseline_expr + 1e-6)
    pred_deg_genes = get_top_genes(pred_fc, top_k=200)
    
    # Enrichment analysis
    actual_pathways = enrichr(actual_deg_genes, gene_sets='KEGG_2021')
    pred_pathways = enrichr(pred_deg_genes, gene_sets='KEGG_2021')
    
    # Compute overlap
    actual_top = set(actual_pathways['Term'][:10])
    pred_top = set(pred_pathways['Term'][:10])
    overlap = len(actual_top & pred_top) / len(actual_top)
    
    return overlap
```

### 6.3 Downstream Task Performance

**Linear probing**:
```python
@torch.no_grad()
def evaluate_linear_probe(model, train_data, train_labels, test_data, test_labels):
    """
    Train linear classifier on embeddings.
    
    Measures quality of learned representations.
    """
    # Extract embeddings
    train_emb = model.encoder(train_data).mean(dim=1)  # Average over tokens
    test_emb = model.encoder(test_data).mean(dim=1)
    
    # Train classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_emb.cpu(), train_labels)
    
    # Evaluate
    accuracy = clf.score(test_emb.cpu(), test_labels)
    
    return accuracy
```

---

## Key Takeaways

### Vision → Biology Mapping

1. **Images → Gene expression** — Patches → Genes/modules
2. **Videos → Time-series** — Frames → Time points
3. **Actions → Perturbations** — Conditioning → Interventions
4. **Multi-view → Multi-omics** — Different views → Different modalities

### Applications

1. **Perturbation prediction** — Most natural JEPA application
2. **Trajectory inference** — Temporal dynamics
3. **Multi-omics** — Cross-modality prediction
4. **Drug response** — Treatment prediction

### Advantages

1. **Efficiency** — No decoder, fast training
2. **Robustness** — Focus on semantics, not pixels
3. **Compositional** — Combine perturbations naturally
4. **Hybrid** — Combine with diffusion for generation

### Best Practices

1. **Start with perturbations** — Most straightforward application
2. **Evaluate on biology** — DEGs, pathways, not just embeddings
3. **Combine with generative** — For uncertainty quantification
4. **Use downstream tasks** — Validate representation quality

---

## Related Documents

- [00_jepa_overview.md](00_jepa_overview.md) — High-level concepts
- [01_jepa_foundations.md](01_jepa_foundations.md) — Architecture details
- [02_jepa_training.md](02_jepa_training.md) — Training strategies
- [04_jepa_perturbseq.md](04_jepa_perturbseq.md) — Detailed Perturb-seq implementation

---

## References

**JEPA papers**:
- Assran et al. (2023): "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
- Bardes et al. (2024): "V-JEPA: Latent Video Prediction"

**Perturbation modeling**:
- Lotfollahi et al. (2019): "scGen predicts single-cell perturbation responses"
- Roohani et al. (2023): "Predicting transcriptional outcomes of novel multigene perturbations with GEARS"
- Norman et al. (2019): "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes"

**Multi-omics**:
- Ma et al. (2020): "Integrative Methods and Practical Challenges for Single-Cell Multi-omics"
- Argelaguet et al. (2021): "MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data"
