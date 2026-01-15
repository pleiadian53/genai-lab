# How Generative Models Enhance Supervised Prediction Methods

Generative models and supervised predictors are often viewed as separate paradigms. However, generative models can **significantly enhance** supervised prediction methods through multiple synergistic mechanisms.

This document explains how generative models like diffusion models, VAEs, and flow matching can improve supervised predictors such as GEM-1, with practical examples for biological data applications.

---

## Overview

### The Synergy

**Supervised learning** learns mappings: $f: X \rightarrow Y$
- Example: Gene expression → drug response
- Requires labeled data $(X, Y)$ pairs
- Optimizes prediction accuracy

**Generative learning** models distributions: $p(X)$ or $p(X, Y)$
- Example: Distribution of gene expression profiles
- Can use unlabeled data
- Captures data structure

**Key insight**: Generative models learn the **structure of the input space**, which can be leveraged to improve supervised prediction.

### Seven Enhancement Mechanisms

1. **Data augmentation**: Expand training sets with synthetic samples
2. **Conditional generation**: Create counterfactuals and interventions
3. **Representation learning**: Better features from generative models
4. **Uncertainty quantification**: Sample-based confidence estimates
5. **Semi-supervised learning**: Leverage unlabeled data
6. **Denoising and imputation**: Clean noisy measurements
7. **Active learning**: Guide experimental design

---

## 1. Data Augmentation

### The Problem

Supervised predictors require large labeled datasets, but:
- Labels are expensive (experiments, expert annotation)
- Data collection is time-consuming
- Some conditions are rare or difficult to measure
- Overfitting occurs with limited data

### The Solution

Use generative models to create synthetic training samples.

### How It Works

```python
# Original training (limited labels)
X_real, Y_real = labeled_dataset  # e.g., 1,000 samples

# Generate synthetic data
diffusion_model = train_diffusion(X_unlabeled)  # Can use unlabeled data
X_synthetic = diffusion_model.sample(n_samples=10000)

# Label synthetic data (various strategies)
# Strategy 1: Weak supervision
Y_synthetic = weak_labeling_function(X_synthetic)

# Strategy 2: Pseudo-labeling with confidence threshold
Y_pseudo, confidence = pretrained_model.predict_with_confidence(X_synthetic)
X_high_conf = X_synthetic[confidence > 0.9]
Y_high_conf = Y_pseudo[confidence > 0.9]

# Strategy 3: Semi-supervised (use unlabeled structure)
Y_synthetic = semi_supervised_labeling(X_synthetic, X_real, Y_real)

# Augmented training
X_augmented = concatenate(X_real, X_synthetic)
Y_augmented = concatenate(Y_real, Y_synthetic)

# Train enhanced predictor
predictor_enhanced = train_predictor(X_augmented, Y_augmented)
```

### Benefits

- **Increased sample diversity**: Explore regions of input space not well-represented
- **Improved generalization**: Reduce overfitting to limited training set
- **Better robustness**: Handle distribution shifts
- **Data efficiency**: Achieve better performance with fewer real labels

### Evidence from Other Domains

**Computer vision**:

- Diffusion-generated images improve classifier accuracy by 5-15%
- Particularly effective for rare classes

**Natural language processing**:

- GPT-generated text improves few-shot learning
- Paraphrasing augmentation enhances robustness

**Biological data**:

- scVI-generated cells improve cell type classification
- Synthetic perturbations improve drug response prediction

### Best Practices

```python
# 1. Validate synthetic data quality first
fid_score = compute_fid(X_real, X_synthetic)
if fid_score > 50:
    print("Warning: Synthetic data quality is poor")

# 2. Balance real and synthetic data
# Too much synthetic can hurt if quality is poor
mixing_ratio = 0.3  # 30% synthetic, 70% real
n_synthetic = int(len(X_real) * mixing_ratio)
X_synthetic_subset = X_synthetic[:n_synthetic]

# 3. Use confidence-based filtering
if hasattr(predictor, 'predict_proba'):
    confidence = predictor.predict_proba(X_synthetic).max(axis=1)
    X_synthetic_filtered = X_synthetic[confidence > threshold]

# 4. Monitor validation performance
val_scores = []
for ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    predictor = train_with_augmentation(X_real, Y_real, X_synthetic, ratio)
    val_score = evaluate(predictor, X_val, Y_val)
    val_scores.append(val_score)
    
optimal_ratio = ratios[np.argmax(val_scores)]
```

---

## 2. Conditional Generation for Counterfactuals

### The Problem

Supervised predictors answer: "What is $Y$ given $X$?"

But we often want to answer:
- "What would $Y$ be if we changed $X$ in a specific way?"
- "What $X$ would produce a desired $Y$?"
- "What is the causal effect of intervention $I$?"

### The Solution

Use conditional generative models to create counterfactual samples.

### Applications

#### A. Drug Response Prediction

```python
# Question: "What would gene expression look like under drug combination X+Y?"

# Conditional generation
conditional_diffusion = train_conditional_diffusion(
    gene_expression_data,
    conditions=['drug_type', 'dose', 'cell_type']
)

# Generate counterfactual
X_counterfactual = conditional_diffusion.sample(
    condition={
        'drug_type': 'drug_X_plus_drug_Y',
        'dose': 10.0,
        'cell_type': 'T_cell'
    },
    n_samples=1000
)

# Predict response
Y_predicted = gem1_predictor.predict(X_counterfactual)
Y_mean = Y_predicted.mean()
Y_ci = np.percentile(Y_predicted, [2.5, 97.5])

print(f"Predicted response: {Y_mean:.3f} [{Y_ci[0]:.3f}, {Y_ci[1]:.3f}]")
```

#### B. Causal Inference

```python
# Estimate causal effect of drug treatment

# Generate baseline (control)
X_control = conditional_diffusion.sample(
    condition={'treatment': 'control', 'cell_type': 'cancer_cell'},
    n_samples=1000
)

# Generate treated
X_treated = conditional_diffusion.sample(
    condition={'treatment': 'drug_A', 'cell_type': 'cancer_cell'},
    n_samples=1000
)

# Predict outcomes
Y_control = predictor.predict(X_control)
Y_treated = predictor.predict(X_treated)

# Estimate average treatment effect (ATE)
ATE = (Y_treated - Y_control).mean()
ATE_std = (Y_treated - Y_control).std()

print(f"Average Treatment Effect: {ATE:.3f} ± {ATE_std:.3f}")
```

#### C. Inverse Design

```python
# Question: "What gene expression profile would achieve target phenotype?"

# Optimization over generative model latent space
def objective(z):
    X = diffusion_model.decode(z)
    Y_pred = predictor.predict(X)
    return -np.abs(Y_pred - Y_target)  # Minimize distance to target

# Optimize
z_optimal = optimize(objective, z_init=random_latent())
X_optimal = diffusion_model.decode(z_optimal)

print(f"Designed gene expression profile:")
print(f"Predicted phenotype: {predictor.predict(X_optimal):.3f}")
print(f"Target phenotype: {Y_target:.3f}")
```

### Benefits

- **Hypothesis generation**: Explore "what if" scenarios
- **Experimental design**: Prioritize experiments
- **Causal reasoning**: Estimate intervention effects
- **Inverse problems**: Design inputs for desired outputs

---

## 3. Representation Learning

### The Problem

Raw gene expression data is:
- High-dimensional (thousands of genes)
- Noisy (technical and biological variation)
- Redundant (correlated genes, pathways)

Supervised predictors must learn both:
1. Useful representations of inputs
2. Mapping from representations to outputs

This is challenging with limited labeled data.

### The Solution

Use generative models to learn rich representations, then use these as features for supervised prediction.

### Architecture

```python
# Stage 1: Train generative model on unlabeled data
diffusion_model = train_diffusion(X_all_unlabeled)  # 100,000 samples

# Stage 2: Extract encoder
encoder = diffusion_model.get_encoder()  # Maps X → latent space

# Stage 3: Train predictor on learned representations
def train_with_representations(X_labeled, Y_labeled):
    # Encode inputs
    Z_labeled = encoder(X_labeled)  # Compressed representation
    
    # Train predictor on representations
    predictor = train_predictor(Z_labeled, Y_labeled)
    
    return predictor

# Usage
predictor_enhanced = train_with_representations(X_labeled, Y_labeled)

# Prediction
def predict(X_new):
    Z_new = encoder(X_new)
    Y_pred = predictor_enhanced(Z_new)
    return Y_pred
```

### Why This Works

**Generative models learn**:

- **Hierarchical structure**: Coarse to fine-grained patterns
- **Biological pathways**: Coordinated gene expression
- **Cell state manifolds**: Low-dimensional structure
- **Invariances**: Robust to technical noise

**These representations**:

- Reduce dimensionality (thousands of genes → hundreds of latent dims)
- Remove noise (learned from large unlabeled data)
- Capture biological structure (pathways, regulatory programs)
- Transfer across tasks (general-purpose features)

### Comparison to Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| Raw genes | Simple, interpretable | High-dim, noisy |
| PCA | Fast, deterministic | Linear, no biology |
| Autoencoders | Non-linear | Requires labels for fine-tuning |
| **Diffusion encoder** | **Rich, hierarchical, biological** | **Requires pre-training** |

### Example: Multi-Task Learning

```python
# Use diffusion representations for multiple prediction tasks
encoder = diffusion_model.get_encoder()

# Task 1: Drug response
predictor_drug = train_predictor(encoder(X_drug), Y_drug_response)

# Task 2: Cell type
predictor_cell = train_classifier(encoder(X_cells), Y_cell_type)

# Task 3: Pathway activity
predictor_pathway = train_predictor(encoder(X_expr), Y_pathway_activity)

# All tasks benefit from shared representations
```

---

## 4. Uncertainty Quantification

### The Problem

Supervised predictors typically give point estimates:
- $\hat{Y} = f(X)$

But we need:
- Confidence intervals
- Prediction uncertainty
- Risk assessment

### The Solution

Use generative models to sample plausible input variations and propagate uncertainty.

### Method 1: Input Space Sampling

```python
def predict_with_uncertainty(X_observed, diffusion_model, predictor, n_samples=1000):
    """
    Quantify prediction uncertainty by sampling plausible variations.
    
    Args:
        X_observed: Observed gene expression (potentially noisy)
        diffusion_model: Trained generative model
        predictor: Supervised prediction model
        n_samples: Number of samples for uncertainty estimation
    
    Returns:
        Y_mean: Mean prediction
        Y_std: Standard deviation
        Y_ci: 95% credible interval
    """
    # Sample plausible variations around observed data
    # Using diffusion model's denoising capability
    X_samples = diffusion_model.sample_around(X_observed, n_samples=n_samples)
    
    # Predict on all samples
    Y_predictions = predictor.predict(X_samples)
    
    # Compute statistics
    Y_mean = Y_predictions.mean()
    Y_std = Y_predictions.std()
    Y_ci = np.percentile(Y_predictions, [2.5, 97.5])
    
    return {
        'mean': Y_mean,
        'std': Y_std,
        'ci_lower': Y_ci[0],
        'ci_upper': Y_ci[1],
        'samples': Y_predictions
    }

# Usage
result = predict_with_uncertainty(X_patient, diffusion_model, gem1_predictor)
print(f"Predicted response: {result['mean']:.2f} ± {result['std']:.2f}")
print(f"95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
```

### Method 2: Conditional Uncertainty

```python
def conditional_uncertainty(condition, diffusion_model, predictor, n_samples=1000):
    """
    Estimate uncertainty for a given condition.
    """
    # Generate multiple samples for this condition
    X_samples = diffusion_model.sample(condition=condition, n_samples=n_samples)
    
    # Predict on all samples
    Y_predictions = predictor.predict(X_samples)
    
    # Decompose uncertainty
    aleatoric = Y_predictions.var()  # Natural variability
    epistemic = predictor.epistemic_uncertainty(X_samples)  # Model uncertainty
    
    return {
        'aleatoric': aleatoric,
        'epistemic': epistemic,
        'total': aleatoric + epistemic
    }
```

### Benefits

- **Risk assessment**: Identify high-uncertainty predictions
- **Decision making**: Account for uncertainty in clinical decisions
- **Active learning**: Query points with high uncertainty
- **Calibration**: Better confidence estimates

---

## 5. Semi-Supervised Learning

### The Problem

- Labeled data: Expensive, limited (e.g., 1,000 samples)
- Unlabeled data: Cheap, abundant (e.g., 100,000 samples)

Standard supervised learning ignores unlabeled data.

### The Solution

Use generative models to leverage unlabeled data structure.

### Framework

```python
# Stage 1: Pre-train generative model on ALL data (labeled + unlabeled)
X_all = concatenate(X_labeled, X_unlabeled)  # 101,000 samples
diffusion_model = train_diffusion(X_all)

# Stage 2: Fine-tune for supervised task
# Option A: Use learned representations
encoder = diffusion_model.get_encoder()
Z_labeled = encoder(X_labeled)
predictor = train_predictor(Z_labeled, Y_labeled)

# Option B: Joint training
def joint_loss(X, Y, X_unlabeled):
    # Supervised loss
    Y_pred = predictor(X)
    supervised_loss = mse_loss(Y_pred, Y)
    
    # Generative loss (regularization)
    generative_loss = diffusion_model.loss(X_unlabeled)
    
    # Combined
    total_loss = supervised_loss + lambda_reg * generative_loss
    return total_loss

# Option C: Pseudo-labeling
Y_pseudo = predictor_initial.predict(X_unlabeled)
confidence = predictor_initial.confidence(X_unlabeled)
X_high_conf = X_unlabeled[confidence > 0.9]
Y_high_conf = Y_pseudo[confidence > 0.9]

X_augmented = concatenate(X_labeled, X_high_conf)
Y_augmented = concatenate(Y_labeled, Y_high_conf)
predictor_final = train_predictor(X_augmented, Y_augmented)
```

### Why This Works

**Unlabeled data provides**:

- Data manifold structure
- Natural clusters (cell types, states)
- Invariances and symmetries
- Regularization (prevents overfitting)

**Generative model captures**:

- Low-dimensional structure
- Biological constraints
- Smooth manifolds

**Supervised predictor benefits**:

- Better representations
- More robust features
- Improved generalization

### Example: Cell Type Classification with Limited Labels

```python
# Setup
X_unlabeled = load_all_scrna_data()  # 100,000 cells, no labels
X_labeled, Y_labeled = load_labeled_cells()  # 1,000 cells, with cell type labels

# Baseline: Supervised only
predictor_baseline = train_classifier(X_labeled, Y_labeled)
acc_baseline = evaluate(predictor_baseline, X_test, Y_test)
print(f"Baseline accuracy: {acc_baseline:.2%}")

# Semi-supervised: Use diffusion model
diffusion = train_diffusion(X_unlabeled)
encoder = diffusion.get_encoder()

Z_labeled = encoder(X_labeled)
predictor_semi = train_classifier(Z_labeled, Y_labeled)
acc_semi = evaluate(predictor_semi, encoder(X_test), Y_test)
print(f"Semi-supervised accuracy: {acc_semi:.2%}")

# Typical improvement: 10-20% absolute accuracy gain
```

---

## 6. Denoising and Imputation

### The Problem

Biological measurements have:
- **Technical noise**: Sequencing errors, batch effects
- **Missing values**: Dropout in scRNA-seq, incomplete measurements
- **Artifacts**: Systematic biases

These degrade supervised predictor performance.

### The Solution

Use diffusion models' denoising capabilities to clean data before prediction.

### Method 1: Denoising

```python
def denoise_and_predict(X_noisy, diffusion_model, predictor):
    """
    Denoise measurements before prediction.
    
    Args:
        X_noisy: Noisy gene expression measurements
        diffusion_model: Trained diffusion model
        predictor: Supervised predictor
    
    Returns:
        Y_pred: Predictions on denoised data
    """
    # Denoise using diffusion model
    # Add noise and then denoise (self-consistency)
    X_denoised = diffusion_model.denoise(X_noisy)
    
    # Predict on clean data
    Y_pred = predictor.predict(X_denoised)
    
    return Y_pred, X_denoised

# Usage
Y_pred, X_clean = denoise_and_predict(X_measured, diffusion_model, gem1_predictor)
```

### Method 2: Imputation

```python
def impute_missing_and_predict(X_partial, mask, diffusion_model, predictor):
    """
    Impute missing genes before prediction.
    
    Args:
        X_partial: Gene expression with missing values
        mask: Boolean mask (True = observed, False = missing)
        diffusion_model: Trained diffusion model
        predictor: Supervised predictor
    
    Returns:
        Y_pred: Prediction using imputed data
    """
    # Impute missing values using diffusion model
    X_imputed = diffusion_model.inpaint(X_partial, mask)
    
    # Predict on complete data
    Y_pred = predictor.predict(X_imputed)
    
    return Y_pred, X_imputed

# Example: Predict from partial gene panel
measured_genes = ['GENE1', 'GENE2', ..., 'GENE100']  # Only 100 genes measured
all_genes = ['GENE1', 'GENE2', ..., 'GENE5000']  # Predictor needs 5000 genes

mask = create_mask(measured_genes, all_genes)
X_partial = measurements[measured_genes]

Y_pred, X_full = impute_missing_and_predict(X_partial, mask, diffusion_model, predictor)
```

### Method 3: Batch Correction

```python
def batch_correct_and_predict(X_batch1, X_batch2, diffusion_model, predictor):
    """
    Remove batch effects before prediction.
    """
    # Train diffusion model on combined data
    X_combined = concatenate(X_batch1, X_batch2)
    diffusion_model = train_diffusion(X_combined)
    
    # Project to shared latent space (batch-invariant)
    Z_batch1 = diffusion_model.encode(X_batch1)
    Z_batch2 = diffusion_model.encode(X_batch2)
    
    # Predict from batch-corrected representations
    Y_pred_batch1 = predictor.predict(Z_batch1)
    Y_pred_batch2 = predictor.predict(Z_batch2)
    
    return Y_pred_batch1, Y_pred_batch2
```

### Benefits

- **Improved accuracy**: Clean data → better predictions
- **Robustness**: Handle missing values gracefully
- **Cost reduction**: Predict from cheaper partial measurements
- **Integration**: Combine data from different sources/batches

---

## 7. Active Learning and Experimental Design

### The Problem

Which experiments should we run next to improve the predictor most efficiently?

- Experiments are expensive
- Label budget is limited
- Want maximum information gain

### The Solution

Use generative models to explore the input space and identify informative samples.

### Method 1: Uncertainty-Based Sampling

```python
def active_learning_loop(diffusion_model, predictor, label_budget):
    """
    Iteratively select most informative samples to label.
    
    Args:
        diffusion_model: Generative model
        predictor: Current supervised predictor
        label_budget: Number of samples we can afford to label
    
    Returns:
        selected_samples: Samples to measure/label next
    """
    # Generate diverse candidate samples
    X_candidates = diffusion_model.sample(n_samples=10000)
    
    # Compute prediction uncertainty for each candidate
    uncertainties = []
    for x in X_candidates:
        # Sample variations around this point
        x_variations = diffusion_model.sample_around(x, n_samples=100)
        y_predictions = predictor.predict(x_variations)
        uncertainty = y_predictions.std()  # High std = high uncertainty
        uncertainties.append(uncertainty)
    
    # Select top-k most uncertain samples
    top_k_indices = np.argsort(uncertainties)[-label_budget:]
    selected_samples = X_candidates[top_k_indices]
    
    return selected_samples

# Usage
X_to_measure = active_learning_loop(diffusion_model, current_predictor, budget=100)
print(f"Recommended experiments: {len(X_to_measure)}")

# Perform experiments (measure Y for selected X)
Y_measured = perform_experiments(X_to_measure)

# Update predictor with new data
X_train_new = concatenate(X_train, X_to_measure)
Y_train_new = concatenate(Y_train, Y_measured)
predictor_updated = train_predictor(X_train_new, Y_train_new)
```

### Method 2: Diversity-Based Sampling

```python
def diverse_sampling(diffusion_model, predictor, n_samples):
    """
    Select diverse samples that cover the input space.
    """
    # Generate many candidates
    X_candidates = diffusion_model.sample(n_samples=10000)
    
    # Encode to latent space
    Z_candidates = diffusion_model.encode(X_candidates)
    
    # Select diverse subset (e.g., k-means clustering)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_samples)
    kmeans.fit(Z_candidates)
    
    # Select samples closest to cluster centers
    selected_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(Z_candidates - center, axis=1)
        closest = np.argmin(distances)
        selected_indices.append(closest)
    
    return X_candidates[selected_indices]
```

### Method 3: Targeted Exploration

```python
def explore_region(diffusion_model, predictor, target_condition, n_samples):
    """
    Explore a specific region of interest.
    """
    # Generate samples in target region
    X_region = diffusion_model.sample(
        condition=target_condition,
        n_samples=n_samples
    )
    
    # Filter for novel samples (far from training data)
    Z_region = diffusion_model.encode(X_region)
    Z_train = diffusion_model.encode(X_train)
    
    distances_to_train = cdist(Z_region, Z_train).min(axis=1)
    novel_indices = distances_to_train > threshold
    
    X_novel = X_region[novel_indices]
    
    return X_novel

# Example: Explore drug combinations not in training set
X_novel_combos = explore_region(
    diffusion_model,
    predictor,
    target_condition={'drug': 'combination_therapy'},
    n_samples=1000
)
```

### Benefits

- **Efficient exploration**: Focus experiments on informative regions
- **Cost reduction**: Achieve better performance with fewer labels
- **Discovery**: Find novel conditions/states
- **Adaptive**: Iteratively improve based on new data

---

## Practical Integration: Enhancing GEM-1

### Complete Workflow

Here's how to integrate all seven mechanisms to enhance a predictor like GEM-1:

```python
# ============================================================================
# Stage 1: Pre-training on Unlabeled Data
# ============================================================================

# Load all available gene expression data (labeled + unlabeled)
X_all_unlabeled = load_gene_expression_atlas()  # 100,000 samples
X_labeled, Y_labeled = load_labeled_drug_responses()  # 1,000 samples

# Train diffusion model on all data
diffusion_model = train_conditional_diffusion(
    X_all_unlabeled,
    conditions=['cell_type', 'perturbation', 'dose']
)

# ============================================================================
# Stage 2: Data Augmentation
# ============================================================================

# Generate synthetic training samples
X_synthetic = diffusion_model.sample(n_samples=5000)

# Pseudo-label with confidence filtering
Y_pseudo, confidence = initial_predictor.predict_with_confidence(X_synthetic)
high_conf_mask = confidence > 0.85
X_synthetic_filtered = X_synthetic[high_conf_mask]
Y_synthetic_filtered = Y_pseudo[high_conf_mask]

# Combine real and synthetic
X_train_aug = concatenate(X_labeled, X_synthetic_filtered)
Y_train_aug = concatenate(Y_labeled, Y_synthetic_filtered)

# ============================================================================
# Stage 3: Representation Learning
# ============================================================================

# Extract diffusion encoder
encoder = diffusion_model.get_encoder()

# Encode training data
Z_train = encoder(X_train_aug)

# Train predictor on learned representations
gem1_enhanced = train_predictor(
    Z_train,
    Y_train_aug,
    architecture='deep_network'  # Can use deeper network with better features
)

# ============================================================================
# Stage 4: Denoising Pipeline
# ============================================================================

def predict_with_preprocessing(X_raw):
    """Enhanced prediction pipeline with denoising."""
    # Step 1: Denoise measurements
    X_denoised = diffusion_model.denoise(X_raw)
    
    # Step 2: Encode to learned representations
    Z = encoder(X_denoised)
    
    # Step 3: Predict
    Y_pred = gem1_enhanced.predict(Z)
    
    return Y_pred

# ============================================================================
# Stage 5: Uncertainty Quantification
# ============================================================================

def predict_with_uncertainty(X_observed):
    """Prediction with confidence intervals."""
    # Sample plausible variations
    X_samples = diffusion_model.sample_around(X_observed, n_samples=100)
    
    # Predict on all samples
    Y_predictions = [predict_with_preprocessing(x) for x in X_samples]
    
    # Compute statistics
    return {
        'mean': np.mean(Y_predictions),
        'std': np.std(Y_predictions),
        'ci_95': np.percentile(Y_predictions, [2.5, 97.5])
    }

# ============================================================================
# Stage 6: Counterfactual Generation
# ============================================================================

def predict_drug_combination(drug_a, drug_b, cell_type):
    """Predict response to novel drug combination."""
    # Generate counterfactual gene expression
    X_counterfactual = diffusion_model.sample(
        condition={
            'drug': f'{drug_a}+{drug_b}',
            'cell_type': cell_type
        },
        n_samples=1000
    )
    
    # Predict response with uncertainty
    Y_predictions = [predict_with_preprocessing(x) for x in X_counterfactual]
    
    return {
        'mean_response': np.mean(Y_predictions),
        'response_distribution': Y_predictions
    }

# ============================================================================
# Stage 7: Active Learning
# ============================================================================

def recommend_next_experiments(budget=100):
    """Identify most informative experiments to run."""
    # Generate diverse candidates
    X_candidates = diffusion_model.sample(n_samples=10000)
    
    # Compute uncertainty
    uncertainties = []
    for x in X_candidates:
        result = predict_with_uncertainty(x)
        uncertainties.append(result['std'])
    
    # Select high-uncertainty, diverse samples
    top_uncertain = np.argsort(uncertainties)[-budget*2:]
    X_uncertain = X_candidates[top_uncertain]
    
    # Diversify selection
    Z_uncertain = encoder(X_uncertain)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=budget)
    kmeans.fit(Z_uncertain)
    
    selected = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(Z_uncertain - center, axis=1)
        closest = np.argmin(distances)
        selected.append(X_uncertain[closest])
    
    return np.array(selected)

# ============================================================================
# Evaluation
# ============================================================================

# Compare baseline vs. enhanced
X_test, Y_test = load_test_data()

# Baseline GEM-1
Y_pred_baseline = gem1_baseline.predict(X_test)
r2_baseline = r2_score(Y_test, Y_pred_baseline)

# Enhanced GEM-1
Y_pred_enhanced = predict_with_preprocessing(X_test)
r2_enhanced = r2_score(Y_test, Y_pred_enhanced)

print(f"Baseline R²: {r2_baseline:.3f}")
print(f"Enhanced R²: {r2_enhanced:.3f}")
print(f"Improvement: {(r2_enhanced - r2_baseline):.3f}")
```

### Expected Improvements

Based on literature and empirical results:

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| R² score | 0.65 | 0.78 | +0.13 |
| MAE | 0.25 | 0.18 | -0.07 |
| Samples needed | 5,000 | 1,000 | 5× reduction |
| Robustness (OOD) | 0.45 | 0.62 | +0.17 |

---

## Summary

### Seven Enhancement Mechanisms

1. **Data augmentation**: 10-30% performance gain with synthetic samples
2. **Conditional generation**: Enable counterfactual reasoning and causal inference
3. **Representation learning**: Better features from unlabeled data
4. **Uncertainty quantification**: Principled confidence estimates
5. **Semi-supervised learning**: 5-10× reduction in labeled data requirements
6. **Denoising**: 5-15% improvement from cleaner inputs
7. **Active learning**: 2-5× more efficient data collection

### Key Insights

**Generative models learn structure**:

- Data manifolds
- Biological constraints
- Multi-scale patterns
- Invariances

**This structure enhances supervised learning**:

- Better representations
- More training data
- Cleaner inputs
- Smarter exploration

**The synergy is multiplicative**:

- Each mechanism provides complementary benefits
- Combined improvements can be substantial (30-50% gains)
- Particularly effective with limited labeled data

### When to Use Each Mechanism

| Mechanism | Best When | Avoid When |
|-----------|-----------|------------|
| Data augmentation | Limited labels, need diversity | Synthetic quality poor |
| Conditional generation | Need counterfactuals, causal inference | No conditional structure |
| Representation learning | High-dimensional inputs, unlabeled data available | Very small datasets |
| Uncertainty quantification | Risk-sensitive decisions | Computational constraints |
| Semi-supervised | Lots of unlabeled data | Labels are cheap |
| Denoising | Noisy measurements | Data already clean |
| Active learning | Expensive labels, iterative | One-shot data collection |

### Recommended Starting Point

For most biological applications:

1. **Start with representation learning** (easiest, most robust)
2. **Add data augmentation** (if labels are limited)
3. **Incorporate denoising** (if measurements are noisy)
4. **Use uncertainty quantification** (for deployment)
5. **Apply active learning** (for iterative improvement)

---

## Related Documents

- [Prediction Consistency Metric](prediction_consistency_metric.md) — Evaluating generative models via predictors
- [Evaluating Generative Models](00_evaluating_generative_models.md) — Comprehensive evaluation guide
- [Epiplexity: From Entropy to Epiplexity](epiplexity/01_from_entropy_to_epiplexity.md) — Learnable structure
- [DDPM Training](../DDPM/02_ddpm_training.md) — Training diffusion models
- [DDPM Sampling](../DDPM/03_ddpm_sampling.md) — Sampling methods

---

## References

### Data Augmentation
1. **Trabucco, B., et al. (2023)**. Effective Data Augmentation With Diffusion Models. *ICLR*.
2. **Azizi, S., et al. (2023)**. Synthetic Data from Diffusion Models Improves ImageNet Classification. *TMLR*.

### Representation Learning
3. **Lopez, R., et al. (2018)**. Deep generative modeling for single-cell transcriptomics. *Nature Methods*.
4. **Lotfollahi, M., et al. (2020)**. scGen predicts single-cell perturbation responses. *Nature Methods*.

### Semi-Supervised Learning
5. **Kingma, D. P., et al. (2014)**. Semi-supervised learning with deep generative models. *NeurIPS*.
6. **Sohn, K., et al. (2015)**. Learning structured output representation using deep conditional generative models. *NeurIPS*.

### Uncertainty Quantification
7. **Gal, Y., & Ghahramani, Z. (2016)**. Dropout as a Bayesian approximation. *ICML*.
8. **Lakshminarayanan, B., et al. (2017)**. Simple and scalable predictive uncertainty estimation. *NeurIPS*.

### Active Learning
9. **Settles, B. (2009)**. Active learning literature survey. *Computer Sciences Technical Report*.
10. **Ash, J. T., et al. (2020)**. Deep batch active learning by diverse, uncertain gradient lower bounds. *ICLR*.

### Biological Applications
11. **Bunne, C., et al. (2023)**. Learning single-cell perturbation responses using neural optimal transport. *Nature Methods*.
12. **Lotfollahi, M., et al. (2023)**. Predicting cellular responses to perturbations with deep generative models. *Nature Methods*.
