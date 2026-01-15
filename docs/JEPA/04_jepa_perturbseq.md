# JEPA for Perturb-seq: Complete Implementation

This document provides a complete, end-to-end implementation of JEPA for Perturb-seq data, from data loading through training, evaluation, and comparison with existing methods.

**Prerequisites**: Understanding of [JEPA foundations](01_jepa_foundations.md), [training](02_jepa_training.md), and [applications](03_jepa_applications.md).

---

## 1. Dataset: Norman et al. (2019)

### 1.1 Dataset Overview

**Norman et al. Perturb-seq dataset**:

- **Cells**: ~100K K562 cells
- **Perturbations**: 101 genes (single and double knockouts)
- **Technology**: CRISPR-based genetic perturbations + scRNA-seq
- **Genes**: ~20K genes measured

**Key features**:

- Single perturbations: 101 genes
- Double perturbations: 20 gene pairs
- Control cells: Non-targeting guides
- Rich phenotypes: Multiple perturbations per gene

### 1.2 Data Loading

```python
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def load_norman_data(data_path='data/norman2019.h5ad'):
    """
    Load Norman et al. Perturb-seq data.
    
    Returns:
        adata: AnnData object with expression and metadata
    """
    # Load data
    adata = sc.read_h5ad(data_path)
    
    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    adata = adata[:, adata.var['highly_variable']]
    
    return adata


def prepare_perturbseq_pairs(adata):
    """
    Prepare baseline-perturbed pairs.
    
    Args:
        adata: AnnData with perturbation metadata
    
    Returns:
        baseline_expr: Baseline expression (control cells)
        perturbed_expr: Perturbed expression
        perturbation_info: Perturbation metadata
    """
    # Get control cells (baseline)
    control_mask = adata.obs['perturbation'] == 'control'
    baseline_cells = adata[control_mask]
    
    # Get perturbed cells
    perturbed_mask = adata.obs['perturbation'] != 'control'
    perturbed_cells = adata[perturbed_mask]
    
    # For each perturbed cell, sample a random control as baseline
    baseline_expr = []
    perturbed_expr = []
    perturbation_info = []
    
    for i in range(len(perturbed_cells)):
        # Random baseline
        baseline_idx = np.random.randint(len(baseline_cells))
        baseline_expr.append(baseline_cells.X[baseline_idx].toarray().flatten())
        
        # Perturbed
        perturbed_expr.append(perturbed_cells.X[i].toarray().flatten())
        
        # Perturbation info
        pert_gene = perturbed_cells.obs['perturbation'].iloc[i]
        perturbation_info.append(pert_gene)
    
    baseline_expr = np.array(baseline_expr)
    perturbed_expr = np.array(perturbed_expr)
    
    return baseline_expr, perturbed_expr, perturbation_info
```

### 1.3 Perturbation Encoding

```python
class PerturbationEncoder:
    """
    Encode perturbation information.
    
    Converts gene names to embeddings.
    """
    def __init__(self, gene_names, embed_dim=128):
        """
        Args:
            gene_names: List of all gene names
            embed_dim: Embedding dimension
        """
        self.gene_names = gene_names
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        self.num_genes = len(gene_names)
        self.embed_dim = embed_dim
        
        # Learnable gene embeddings
        self.gene_embeddings = nn.Embedding(self.num_genes, embed_dim)
    
    def encode(self, perturbation_list):
        """
        Encode list of perturbations.
        
        Args:
            perturbation_list: List of perturbation strings
                e.g., ['MAPK1', 'MAPK1+BRAF', 'control']
        
        Returns:
            embeddings: Perturbation embeddings (B, embed_dim)
        """
        embeddings = []
        
        for pert in perturbation_list:
            if pert == 'control':
                # Zero embedding for control
                emb = torch.zeros(self.embed_dim)
            elif '+' in pert:
                # Double perturbation: average embeddings
                genes = pert.split('+')
                gene_indices = [self.gene_to_idx[g] for g in genes if g in self.gene_to_idx]
                if gene_indices:
                    embs = self.gene_embeddings(torch.tensor(gene_indices))
                    emb = embs.mean(dim=0)
                else:
                    emb = torch.zeros(self.embed_dim)
            else:
                # Single perturbation
                if pert in self.gene_to_idx:
                    gene_idx = self.gene_to_idx[pert]
                    emb = self.gene_embeddings(torch.tensor(gene_idx))
                else:
                    emb = torch.zeros(self.embed_dim)
            
            embeddings.append(emb)
        
        return torch.stack(embeddings)
```

### 1.4 Dataset Class

```python
class NormanPerturbSeqDataset(Dataset):
    """
    Dataset for Norman Perturb-seq with JEPA.
    """
    def __init__(
        self,
        baseline_expr,
        perturbed_expr,
        perturbation_info,
        perturbation_encoder,
    ):
        """
        Args:
            baseline_expr: Baseline expression (N, num_genes)
            perturbed_expr: Perturbed expression (N, num_genes)
            perturbation_info: List of perturbation strings
            perturbation_encoder: PerturbationEncoder instance
        """
        self.baseline_expr = torch.tensor(baseline_expr, dtype=torch.float32)
        self.perturbed_expr = torch.tensor(perturbed_expr, dtype=torch.float32)
        self.perturbation_info = perturbation_info
        self.perturbation_encoder = perturbation_encoder
    
    def __len__(self):
        return len(self.baseline_expr)
    
    def __getitem__(self, idx):
        baseline = self.baseline_expr[idx]
        perturbed = self.perturbed_expr[idx]
        pert_info = self.perturbation_info[idx]
        
        # Encode perturbation
        pert_emb = self.perturbation_encoder.encode([pert_info])[0]
        
        return baseline, perturbed, pert_emb
```

---

## 2. Model Architecture

### 2.1 Complete JEPA Model for Perturb-seq

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbSeqJEPA(nn.Module):
    """
    Complete JEPA model for Perturb-seq.
    
    Predicts perturbed cell state from baseline + perturbation.
    """
    def __init__(
        self,
        num_genes=5000,
        embed_dim=256,
        num_tokens=64,
        encoder_depth=6,
        predictor_depth=4,
        num_heads=8,
        perturbation_dim=128,
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        
        # Gene expression encoder
        self.encoder = GeneExpressionEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            hidden_dims=[2048, 1024],
            num_tokens=num_tokens,
        )
        
        # Perturbation encoder (learnable)
        self.perturbation_encoder = PerturbationEncoder(
            gene_names=None,  # Will be set later
            embed_dim=perturbation_dim,
        )
        
        # Conditional predictor
        self.predictor = ConditionalPredictor(
            embed_dim=embed_dim,
            condition_dim=perturbation_dim,
            depth=predictor_depth,
            num_heads=num_heads,
        )
        
        # VICReg loss
        self.vicreg = VICRegLoss(
            lambda_inv=25.0,
            lambda_var=25.0,
            lambda_cov=1.0,
        )
    
    def forward(self, x_baseline, x_perturbed, pert_emb):
        """
        Forward pass.
        
        Args:
            x_baseline: Baseline expression (B, num_genes)
            x_perturbed: Perturbed expression (B, num_genes)
            pert_emb: Perturbation embedding (B, perturbation_dim)
        
        Returns:
            loss: Total loss
            loss_dict: Loss components
        """
        # Encode baseline and perturbed
        z_baseline = self.encoder(x_baseline)
        
        with torch.no_grad():
            z_perturbed = self.encoder(x_perturbed)
        
        # Predict perturbed from baseline + perturbation
        z_pred = self.predictor(z_baseline, pert_emb)
        
        # VICReg loss
        loss, loss_dict = self.vicreg(z_pred, z_perturbed)
        
        return loss, loss_dict
    
    @torch.no_grad()
    def predict(self, x_baseline, pert_emb):
        """
        Predict perturbed state.
        
        Args:
            x_baseline: Baseline expression (B, num_genes)
            pert_emb: Perturbation embedding (B, perturbation_dim)
        
        Returns:
            z_pred: Predicted perturbed embedding (B, num_tokens, embed_dim)
        """
        z_baseline = self.encoder(x_baseline)
        z_pred = self.predictor(z_baseline, pert_emb)
        return z_pred
```

---

## 3. Training

### 3.1 Training Script

```python
def train_perturbseq_jepa(
    data_path='data/norman2019.h5ad',
    save_dir='checkpoints/perturbseq_jepa',
    num_epochs=100,
    batch_size=64,
    lr=1e-3,
    device='cuda',
):
    """
    Train JEPA on Perturb-seq data.
    
    Args:
        data_path: Path to Norman data
        save_dir: Checkpoint directory
        num_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device
    """
    # Load data
    print("Loading data...")
    adata = load_norman_data(data_path)
    baseline_expr, perturbed_expr, pert_info = prepare_perturbseq_pairs(adata)
    
    # Split train/val/test
    n_samples = len(baseline_expr)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Create perturbation encoder
    gene_names = adata.var_names.tolist()
    pert_encoder = PerturbationEncoder(gene_names, embed_dim=128)
    
    # Create datasets
    train_dataset = NormanPerturbSeqDataset(
        baseline_expr[train_idx],
        perturbed_expr[train_idx],
        [pert_info[i] for i in train_idx],
        pert_encoder,
    )
    
    val_dataset = NormanPerturbSeqDataset(
        baseline_expr[val_idx],
        perturbed_expr[val_idx],
        [pert_info[i] for i in val_idx],
        pert_encoder,
    )
    
    test_dataset = NormanPerturbSeqDataset(
        baseline_expr[test_idx],
        perturbed_expr[test_idx],
        [pert_info[i] for i in test_idx],
        pert_encoder,
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    print("Creating model...")
    model = PerturbSeqJEPA(
        num_genes=adata.n_vars,
        embed_dim=256,
        num_tokens=64,
        encoder_depth=6,
        predictor_depth=4,
    )
    model.perturbation_encoder = pert_encoder
    model.to(device)
    
    # Train
    print("Training...")
    train_jepa_complete(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=0.01,
        warmup_epochs=10,
        device=device,
        save_dir=save_dir,
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_metrics = evaluate_perturbseq(model, test_loader, device)
    print(f"Test embedding similarity: {test_metrics['embedding_similarity']:.4f}")
    
    return model, test_metrics


# Run training
if __name__ == '__main__':
    model, metrics = train_perturbseq_jepa(
        data_path='data/norman2019.h5ad',
        num_epochs=100,
        batch_size=64,
        lr=1e-3,
    )
```

### 3.2 Hyperparameters

**Recommended settings for Norman data**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Batch size** | 64 | Adjust based on GPU memory |
| **Learning rate** | 1e-3 | Higher than images |
| **Embed dim** | 256 | Balance capacity and speed |
| **Num tokens** | 64 | Compress 5K genes to 64 tokens |
| **Encoder depth** | 6 | Moderate depth |
| **Predictor depth** | 4 | 0.67× encoder depth |
| **Warmup epochs** | 10 | ~10% of total |
| **Weight decay** | 0.01 | Regularization |

---

## 4. Evaluation

### 4.1 Embedding-Level Metrics

```python
@torch.no_grad()
def evaluate_perturbseq(model, test_loader, device):
    """
    Evaluate JEPA on Perturb-seq test set.
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_similarities = []
    all_distances = []
    
    for x_baseline, x_perturbed, pert_emb in test_loader:
        x_baseline = x_baseline.to(device)
        x_perturbed = x_perturbed.to(device)
        pert_emb = pert_emb.to(device)
        
        # Predict
        z_pred = model.predict(x_baseline, pert_emb)
        
        # Actual
        z_actual = model.encoder(x_perturbed)
        
        # Average over tokens
        z_pred_mean = z_pred.mean(dim=1)  # (B, embed_dim)
        z_actual_mean = z_actual.mean(dim=1)
        
        # Cosine similarity
        similarity = F.cosine_similarity(z_pred_mean, z_actual_mean, dim=1)
        all_similarities.append(similarity.cpu())
        
        # L2 distance
        distance = torch.norm(z_pred_mean - z_actual_mean, dim=1)
        all_distances.append(distance.cpu())
    
    # Aggregate
    all_similarities = torch.cat(all_similarities)
    all_distances = torch.cat(all_distances)
    
    metrics = {
        'embedding_similarity': all_similarities.mean().item(),
        'embedding_similarity_std': all_similarities.std().item(),
        'embedding_distance': all_distances.mean().item(),
        'embedding_distance_std': all_distances.std().item(),
    }
    
    return metrics
```

### 4.2 Held-Out Perturbation Evaluation

```python
def evaluate_held_out_perturbations(
    model,
    adata,
    held_out_genes,
    device='cuda',
):
    """
    Evaluate on held-out perturbations.
    
    Tests generalization to unseen perturbations.
    
    Args:
        model: Trained JEPA model
        adata: Full AnnData
        held_out_genes: List of genes to hold out
        device: Device
    
    Returns:
        metrics: Evaluation metrics on held-out perturbations
    """
    model.eval()
    
    # Get cells with held-out perturbations
    held_out_mask = adata.obs['perturbation'].isin(held_out_genes)
    held_out_cells = adata[held_out_mask]
    
    # Get control cells
    control_mask = adata.obs['perturbation'] == 'control'
    control_cells = adata[control_mask]
    
    # Prepare pairs
    baseline_expr = []
    perturbed_expr = []
    pert_info = []
    
    for i in range(len(held_out_cells)):
        baseline_idx = np.random.randint(len(control_cells))
        baseline_expr.append(control_cells.X[baseline_idx].toarray().flatten())
        perturbed_expr.append(held_out_cells.X[i].toarray().flatten())
        pert_info.append(held_out_cells.obs['perturbation'].iloc[i])
    
    baseline_expr = torch.tensor(np.array(baseline_expr), dtype=torch.float32).to(device)
    perturbed_expr = torch.tensor(np.array(perturbed_expr), dtype=torch.float32).to(device)
    
    # Encode perturbations
    pert_embs = model.perturbation_encoder.encode(pert_info).to(device)
    
    # Predict
    z_pred = model.predict(baseline_expr, pert_embs)
    z_actual = model.encoder(perturbed_expr)
    
    # Metrics
    z_pred_mean = z_pred.mean(dim=1)
    z_actual_mean = z_actual.mean(dim=1)
    
    similarity = F.cosine_similarity(z_pred_mean, z_actual_mean, dim=1).mean().item()
    
    print(f"Held-out perturbations ({len(held_out_genes)} genes):")
    print(f"  Embedding similarity: {similarity:.4f}")
    
    return {'held_out_similarity': similarity}
```

### 4.3 Comparison with Baselines

```python
def compare_with_baselines(
    jepa_model,
    test_loader,
    device='cuda',
):
    """
    Compare JEPA with baseline methods.
    
    Baselines:
    1. Mean prediction (predict mean of perturbed cells)
    2. Baseline copy (no change from baseline)
    3. scGen (if available)
    
    Args:
        jepa_model: Trained JEPA model
        test_loader: Test data loader
        device: Device
    
    Returns:
        comparison: Dictionary with results for each method
    """
    jepa_model.eval()
    
    # Collect all data
    all_baseline = []
    all_perturbed = []
    all_pert_emb = []
    
    for x_baseline, x_perturbed, pert_emb in test_loader:
        all_baseline.append(x_baseline)
        all_perturbed.append(x_perturbed)
        all_pert_emb.append(pert_emb)
    
    all_baseline = torch.cat(all_baseline, dim=0).to(device)
    all_perturbed = torch.cat(all_perturbed, dim=0).to(device)
    all_pert_emb = torch.cat(all_pert_emb, dim=0).to(device)
    
    # 1. JEPA
    z_pred_jepa = jepa_model.predict(all_baseline, all_pert_emb)
    z_actual = jepa_model.encoder(all_perturbed)
    
    z_pred_jepa_mean = z_pred_jepa.mean(dim=1)
    z_actual_mean = z_actual.mean(dim=1)
    
    jepa_similarity = F.cosine_similarity(z_pred_jepa_mean, z_actual_mean, dim=1).mean().item()
    
    # 2. Mean prediction (predict mean of all perturbed)
    z_mean_pred = z_actual_mean.mean(dim=0, keepdim=True).repeat(len(z_actual_mean), 1)
    mean_similarity = F.cosine_similarity(z_mean_pred, z_actual_mean, dim=1).mean().item()
    
    # 3. Baseline copy (no change)
    z_baseline = jepa_model.encoder(all_baseline).mean(dim=1)
    baseline_similarity = F.cosine_similarity(z_baseline, z_actual_mean, dim=1).mean().item()
    
    # Results
    comparison = {
        'JEPA': jepa_similarity,
        'Mean prediction': mean_similarity,
        'Baseline copy': baseline_similarity,
    }
    
    print("\nComparison with baselines:")
    for method, sim in comparison.items():
        print(f"  {method}: {sim:.4f}")
    
    return comparison
```

---

## 5. Analysis and Visualization

### 5.1 Embedding Space Visualization

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

@torch.no_grad()
def visualize_embeddings(model, test_loader, device='cuda'):
    """
    Visualize predicted vs actual embeddings.
    
    Args:
        model: Trained JEPA model
        test_loader: Test data loader
        device: Device
    """
    model.eval()
    
    # Collect embeddings
    z_pred_list = []
    z_actual_list = []
    pert_list = []
    
    for x_baseline, x_perturbed, pert_emb in test_loader:
        x_baseline = x_baseline.to(device)
        x_perturbed = x_perturbed.to(device)
        pert_emb = pert_emb.to(device)
        
        z_pred = model.predict(x_baseline, pert_emb).mean(dim=1)
        z_actual = model.encoder(x_perturbed).mean(dim=1)
        
        z_pred_list.append(z_pred.cpu())
        z_actual_list.append(z_actual.cpu())
    
    z_pred_all = torch.cat(z_pred_list, dim=0).numpy()
    z_actual_all = torch.cat(z_actual_list, dim=0).numpy()
    
    # PCA
    pca = PCA(n_components=2)
    z_combined = np.vstack([z_pred_all, z_actual_all])
    z_pca = pca.fit_transform(z_combined)
    
    z_pred_pca = z_pca[:len(z_pred_all)]
    z_actual_pca = z_pca[len(z_pred_all):]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.scatter(z_pred_pca[:, 0], z_pred_pca[:, 1], 
               alpha=0.5, label='Predicted', s=10)
    ax.scatter(z_actual_pca[:, 0], z_actual_pca[:, 1], 
               alpha=0.5, label='Actual', s=10)
    
    # Draw lines connecting pairs
    for i in range(min(100, len(z_pred_pca))):
        ax.plot([z_pred_pca[i, 0], z_actual_pca[i, 0]],
                [z_pred_pca[i, 1], z_actual_pca[i, 1]],
                'k-', alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.set_title('Predicted vs Actual Embeddings')
    
    plt.tight_layout()
    plt.savefig('embeddings_visualization.png', dpi=300)
    plt.show()
```

### 5.2 Per-Perturbation Analysis

```python
@torch.no_grad()
def analyze_per_perturbation(model, adata, device='cuda'):
    """
    Analyze prediction quality per perturbation.
    
    Args:
        model: Trained JEPA model
        adata: AnnData with all data
        device: Device
    
    Returns:
        results: DataFrame with per-perturbation metrics
    """
    model.eval()
    
    # Get unique perturbations
    perturbations = adata.obs['perturbation'].unique()
    perturbations = [p for p in perturbations if p != 'control']
    
    results = []
    
    for pert in perturbations:
        # Get cells with this perturbation
        pert_mask = adata.obs['perturbation'] == pert
        pert_cells = adata[pert_mask]
        
        if len(pert_cells) < 10:
            continue
        
        # Get control cells
        control_mask = adata.obs['perturbation'] == 'control'
        control_cells = adata[control_mask]
        
        # Prepare data
        baseline_expr = []
        perturbed_expr = []
        
        for i in range(len(pert_cells)):
            baseline_idx = np.random.randint(len(control_cells))
            baseline_expr.append(control_cells.X[baseline_idx].toarray().flatten())
            perturbed_expr.append(pert_cells.X[i].toarray().flatten())
        
        baseline_expr = torch.tensor(np.array(baseline_expr), dtype=torch.float32).to(device)
        perturbed_expr = torch.tensor(np.array(perturbed_expr), dtype=torch.float32).to(device)
        
        # Encode perturbation
        pert_emb = model.perturbation_encoder.encode([pert] * len(baseline_expr)).to(device)
        
        # Predict
        z_pred = model.predict(baseline_expr, pert_emb).mean(dim=1)
        z_actual = model.encoder(perturbed_expr).mean(dim=1)
        
        # Metrics
        similarity = F.cosine_similarity(z_pred, z_actual, dim=1).mean().item()
        distance = torch.norm(z_pred - z_actual, dim=1).mean().item()
        
        results.append({
            'perturbation': pert,
            'n_cells': len(pert_cells),
            'similarity': similarity,
            'distance': distance,
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('similarity', ascending=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Similarity
    axes[0].barh(range(len(results_df)), results_df['similarity'])
    axes[0].set_yticks(range(len(results_df)))
    axes[0].set_yticklabels(results_df['perturbation'], fontsize=6)
    axes[0].set_xlabel('Embedding Similarity')
    axes[0].set_title('Prediction Quality per Perturbation')
    
    # Distance
    axes[1].barh(range(len(results_df)), results_df['distance'])
    axes[1].set_yticks(range(len(results_df)))
    axes[1].set_yticklabels(results_df['perturbation'], fontsize=6)
    axes[1].set_xlabel('Embedding Distance')
    axes[1].set_title('Prediction Error per Perturbation')
    
    plt.tight_layout()
    plt.savefig('per_perturbation_analysis.png', dpi=300)
    plt.show()
    
    return results_df
```

---

## 6. Downstream Applications

### 6.1 Virtual Screening

```python
@torch.no_grad()
def virtual_screen_perturbations(
    model,
    baseline_cells,
    candidate_perturbations,
    target_phenotype,
    device='cuda',
):
    """
    Screen candidate perturbations for desired phenotype.
    
    Args:
        model: Trained JEPA model
        baseline_cells: Baseline expression (N, num_genes)
        candidate_perturbations: List of perturbation names
        target_phenotype: Target embedding (embed_dim,)
        device: Device
    
    Returns:
        rankings: Perturbations ranked by similarity to target
    """
    model.eval()
    
    baseline_cells = torch.tensor(baseline_cells, dtype=torch.float32).to(device)
    target_phenotype = target_phenotype.to(device)
    
    results = []
    
    for pert in candidate_perturbations:
        # Encode perturbation
        pert_emb = model.perturbation_encoder.encode([pert] * len(baseline_cells)).to(device)
        
        # Predict
        z_pred = model.predict(baseline_cells, pert_emb).mean(dim=1)  # (N, embed_dim)
        
        # Average over cells
        z_pred_mean = z_pred.mean(dim=0)  # (embed_dim,)
        
        # Similarity to target
        similarity = F.cosine_similarity(z_pred_mean.unsqueeze(0), target_phenotype.unsqueeze(0)).item()
        
        results.append({
            'perturbation': pert,
            'similarity_to_target': similarity,
        })
    
    # Rank by similarity
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('similarity_to_target', ascending=False)
    
    print("Top 10 perturbations for target phenotype:")
    print(results_df.head(10))
    
    return results_df
```

### 6.2 Combination Prediction

```python
@torch.no_grad()
def predict_combination(
    model,
    baseline_cells,
    gene1,
    gene2,
    device='cuda',
):
    """
    Predict effect of double perturbation.
    
    Args:
        model: Trained JEPA model
        baseline_cells: Baseline expression (N, num_genes)
        gene1: First gene to perturb
        gene2: Second gene to perturb
        device: Device
    
    Returns:
        z_pred: Predicted embedding for combination
    """
    model.eval()
    
    baseline_cells = torch.tensor(baseline_cells, dtype=torch.float32).to(device)
    
    # Encode combination
    combination_name = f"{gene1}+{gene2}"
    pert_emb = model.perturbation_encoder.encode([combination_name] * len(baseline_cells)).to(device)
    
    # Predict
    z_pred = model.predict(baseline_cells, pert_emb)
    
    return z_pred
```

---

## Key Takeaways

### Dataset

1. **Norman et al.** — Standard Perturb-seq benchmark
2. **101 genes** — Single and double perturbations
3. **~100K cells** — Rich dataset for training
4. **5K genes** — Use highly variable genes

### Model

1. **Gene expression encoder** — MLP + transformer on tokens
2. **Perturbation encoder** — Learnable gene embeddings
3. **Conditional predictor** — Cross-attention with perturbation
4. **VICReg loss** — Prevents collapse

### Training

1. **Batch size 64** — Balance speed and memory
2. **LR 1e-3** — Higher than images
3. **100 epochs** — Sufficient for convergence
4. **Warmup 10 epochs** — Stabilize early training

### Evaluation

1. **Embedding similarity** — Primary metric
2. **Held-out perturbations** — Test generalization
3. **Per-perturbation analysis** — Identify strengths/weaknesses
4. **Comparison with baselines** — Validate improvements

### Applications

1. **Virtual screening** — Predict effects of new perturbations
2. **Combination prediction** — Double perturbations
3. **Phenotype search** — Find perturbations for target state
4. **Mechanism discovery** — Analyze learned representations

---

## Related Documents

- [00_jepa_overview.md](00_jepa_overview.md) — High-level concepts
- [01_jepa_foundations.md](01_jepa_foundations.md) — Architecture details
- [02_jepa_training.md](02_jepa_training.md) — Training strategies
- [03_jepa_applications.md](03_jepa_applications.md) — General applications

---

## References

**Perturb-seq data**:

- Norman et al. (2019): "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes"
- Replogle et al. (2022): "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq"

**Baseline methods**:

- Lotfollahi et al. (2019): "scGen predicts single-cell perturbation responses"
- Roohani et al. (2023): "Predicting transcriptional outcomes of novel multigene perturbations with GEARS"

**JEPA**:

- Assran et al. (2023): "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
