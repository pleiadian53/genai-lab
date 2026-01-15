# Prediction Consistency: Evaluating Generative Models via Downstream Predictors

A critical question when evaluating generative models for biological data is: **Does the generated data obey the same biological relationships that exist in real data?**

Prediction consistency provides a principled way to answer this question by testing whether generated samples are consistent with learned biological relationships captured by supervised prediction models.

This document explains the theory, implementation, and best practices for using prediction consistency as an evaluation metric for generative models.

---

## Overview

### The Core Idea

**Prediction consistency** measures whether samples from a generative model are compatible with biological relationships learned by a supervised predictor.

**Setup**:
1. Train a supervised predictor on real data: $f: X \rightarrow Y$
   - Example: Gene expression → drug response (like GEM-1)
2. Generate synthetic samples: $\tilde{X} \sim p_\theta(\text{data})$
3. Predict on synthetic samples: $\tilde{Y} = f(\tilde{X})$
4. Compare predictions to expected biological behavior

**Key insight**: If the generative model captures true biological structure, its samples should produce predictions that are consistent with known biology.

### Why This Matters

Traditional evaluation metrics focus on statistical properties:
- **FID**: Distribution distance in feature space
- **Correlation**: Gene-gene relationships
- **Marginals**: Per-gene statistics

**But these don't directly test**:

- Does generated data obey biological laws?
- Would generated data be useful for downstream tasks?
- Does it capture causal relationships?

**Prediction consistency bridges this gap** by testing biological validity through learned relationships.

---

## Theoretical Foundation

### What Prediction Consistency Measures

Given:
- Real data distribution: $p_{\text{real}}(X, Y)$
- Generative model: $p_\theta(X)$
- Learned predictor: $f: X \rightarrow Y$

**Prediction consistency tests**:

$$
p_\theta(Y | f) \stackrel{?}{\approx} p_{\text{real}}(Y)
$$

where $p_\theta(Y | f)$ is the distribution of predictions when applying $f$ to samples from $p_\theta(X)$.

### Interpretation

**High consistency** means:
- Generated samples produce predictions similar to real data
- Generative model respects learned biological relationships
- Samples are likely biologically plausible

**Low consistency** means:
- Generated samples violate learned relationships
- Model may be memorizing noise without capturing biology
- Samples may be statistically correct but biologically invalid

### Connection to Conditional Distributions

For conditional generation, we can test:

$$
p_\theta(Y | X, c) \stackrel{?}{\approx} p_{\text{real}}(Y | c)
$$

where $c$ is a condition (e.g., cell type, perturbation).

This tests whether the generative model correctly captures **conditional relationships**.

---

## Implementation Approaches

### 1. Distribution Consistency

**Objective**: Compare the distribution of predictions on real vs. generated data.

**Method**:

```python
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp

def distribution_consistency(predictor, X_real, X_gen):
    """
    Measure consistency of prediction distributions.
    
    Args:
        predictor: Trained prediction model (e.g., GEM-1)
        X_real: Real gene expression data [N, genes]
        X_gen: Generated gene expression data [M, genes]
    
    Returns:
        metrics: Dictionary of consistency metrics
    """
    # Get predictions
    Y_pred_real = predictor.predict(X_real)
    Y_pred_gen = predictor.predict(X_gen)
    
    # Compute distribution distances
    metrics = {}
    
    # Wasserstein distance (Earth Mover's Distance)
    metrics['wasserstein'] = wasserstein_distance(
        Y_pred_real.flatten(), 
        Y_pred_gen.flatten()
    )
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = ks_2samp(
        Y_pred_real.flatten(), 
        Y_pred_gen.flatten()
    )
    metrics['ks_statistic'] = ks_stat
    metrics['ks_pvalue'] = ks_pval
    
    # Mean and variance comparison
    metrics['mean_diff'] = np.abs(
        Y_pred_real.mean() - Y_pred_gen.mean()
    )
    metrics['var_ratio'] = (
        Y_pred_gen.var() / Y_pred_real.var()
    )
    
    return metrics
```

**Interpretation**:

- **Wasserstein distance**: Lower is better (0 = identical distributions)
- **KS p-value**: Higher is better (>0.05 suggests distributions are similar)
- **Mean difference**: Should be small
- **Variance ratio**: Should be close to 1

### 2. Conditional Consistency

**Objective**: Test if conditional generation produces appropriate predictions.

**Method**:

```python
def conditional_consistency(predictor, gen_model, conditions, Y_expected):
    """
    Test if conditional generation produces expected predictions.
    
    Args:
        predictor: Trained prediction model
        gen_model: Conditional generative model
        conditions: List of conditions to test
        Y_expected: Expected prediction ranges for each condition
    
    Returns:
        consistency_scores: Per-condition consistency
    """
    consistency_scores = {}
    
    for condition in conditions:
        # Generate samples for this condition
        X_gen = gen_model.sample(
            condition=condition, 
            n_samples=1000
        )
        
        # Predict
        Y_pred = predictor.predict(X_gen)
        
        # Check if predictions match expected range
        Y_exp_mean, Y_exp_std = Y_expected[condition]
        
        # Z-score of mean prediction
        z_score = abs(Y_pred.mean() - Y_exp_mean) / Y_exp_std
        
        # Fraction of predictions in expected range
        in_range = np.mean(
            (Y_pred >= Y_exp_mean - 2*Y_exp_std) & 
            (Y_pred <= Y_exp_mean + 2*Y_exp_std)
        )
        
        consistency_scores[condition] = {
            'z_score': z_score,
            'fraction_in_range': in_range,
            'mean_prediction': Y_pred.mean(),
            'std_prediction': Y_pred.std()
        }
    
    return consistency_scores
```

### 3. Perturbation Response Consistency

**Objective**: Test if generated perturbation responses match known biology.

**Method**:

```python
def perturbation_consistency(predictor, gen_model, perturbations):
    """
    Test if perturbation responses are biologically consistent.
    
    Args:
        predictor: Trained prediction model
        gen_model: Conditional generative model
        perturbations: List of (baseline, perturbed) condition pairs
    
    Returns:
        response_metrics: Perturbation response consistency
    """
    response_metrics = {}
    
    for baseline_cond, perturbed_cond in perturbations:
        # Generate baseline and perturbed samples
        X_baseline = gen_model.sample(condition=baseline_cond, n_samples=1000)
        X_perturbed = gen_model.sample(condition=perturbed_cond, n_samples=1000)
        
        # Predict responses
        Y_baseline = predictor.predict(X_baseline)
        Y_perturbed = predictor.predict(X_perturbed)
        
        # Compute response
        delta_Y = Y_perturbed - Y_baseline
        
        # Compare to known perturbation effect
        known_effect = get_known_perturbation_effect(
            baseline_cond, 
            perturbed_cond
        )
        
        # Metrics
        response_metrics[f"{baseline_cond}_to_{perturbed_cond}"] = {
            'mean_response': delta_Y.mean(),
            'expected_response': known_effect['mean'],
            'response_error': abs(delta_Y.mean() - known_effect['mean']),
            'direction_correct': np.sign(delta_Y.mean()) == np.sign(known_effect['mean']),
            'effect_size_ratio': delta_Y.mean() / known_effect['mean']
        }
    
    return response_metrics
```

### 4. Multi-Predictor Ensemble Consistency

**Objective**: Use multiple predictors to avoid dependence on a single model.

**Method**:

```python
def ensemble_consistency(predictors, X_real, X_gen):
    """
    Measure consistency across multiple predictors.
    
    Args:
        predictors: List of trained prediction models
        X_real: Real data
        X_gen: Generated data
    
    Returns:
        ensemble_metrics: Aggregated consistency scores
    """
    consistency_scores = []
    
    for predictor in predictors:
        # Get predictions
        Y_real = predictor.predict(X_real)
        Y_gen = predictor.predict(X_gen)
        
        # Compute consistency for this predictor
        score = wasserstein_distance(Y_real.flatten(), Y_gen.flatten())
        consistency_scores.append(score)
    
    ensemble_metrics = {
        'mean_consistency': np.mean(consistency_scores),
        'std_consistency': np.std(consistency_scores),
        'min_consistency': np.min(consistency_scores),
        'max_consistency': np.max(consistency_scores),
        'per_predictor': consistency_scores
    }
    
    return ensemble_metrics
```

---

## Practical Examples

### Example 1: Drug Response Prediction

**Scenario**: Evaluating a diffusion model that generates gene expression data.

```python
# Setup
gem1_predictor = load_pretrained_gem1()  # Predicts drug response
diffusion_model = train_diffusion_model(gene_expression_data)

# Real data
X_real, Y_real = load_test_data()

# Generated data
X_gen = diffusion_model.sample(n_samples=10000)

# Evaluate prediction consistency
consistency = distribution_consistency(
    predictor=gem1_predictor,
    X_real=X_real,
    X_gen=X_gen
)

print(f"Wasserstein distance: {consistency['wasserstein']:.4f}")
print(f"KS p-value: {consistency['ks_pvalue']:.4f}")
print(f"Mean difference: {consistency['mean_diff']:.4f}")
```

**Interpretation**:

- Wasserstein < 0.1: Excellent consistency
- KS p-value > 0.05: Distributions statistically similar
- Mean difference < 0.05: Predictions well-calibrated

### Example 2: Cell Type-Specific Generation

**Scenario**: Testing conditional generation of cell type-specific gene expression.

```python
# Conditional diffusion model
cond_diffusion = train_conditional_diffusion(
    gene_expression_data,
    conditions=cell_types
)

# Expected predictions for each cell type
Y_expected = {
    'T_cell': (0.8, 0.1),      # (mean, std) for T cell marker
    'B_cell': (0.2, 0.05),     # Low T cell marker for B cells
    'Macrophage': (0.1, 0.05)  # Low T cell marker for macrophages
}

# Test consistency
consistency = conditional_consistency(
    predictor=tcell_marker_predictor,
    gen_model=cond_diffusion,
    conditions=['T_cell', 'B_cell', 'Macrophage'],
    Y_expected=Y_expected
)

for cell_type, metrics in consistency.items():
    print(f"{cell_type}:")
    print(f"  Z-score: {metrics['z_score']:.2f}")
    print(f"  Fraction in range: {metrics['fraction_in_range']:.2%}")
```

### Example 3: Perturbation Response

**Scenario**: Validating that generated perturbation responses match known biology.

```python
# Perturbations to test
perturbations = [
    ('control', 'drug_A'),
    ('control', 'drug_B'),
    ('drug_A', 'drug_A_plus_B')
]

# Test perturbation consistency
response_metrics = perturbation_consistency(
    predictor=response_predictor,
    gen_model=conditional_diffusion,
    perturbations=perturbations
)

for pert, metrics in response_metrics.items():
    print(f"\n{pert}:")
    print(f"  Mean response: {metrics['mean_response']:.3f}")
    print(f"  Expected: {metrics['expected_response']:.3f}")
    print(f"  Direction correct: {metrics['direction_correct']}")
    print(f"  Effect size ratio: {metrics['effect_size_ratio']:.2f}")
```

---

## Advantages

### 1. Biologically Grounded

Tests whether generated data obeys **learned biological relationships**, not just statistical properties.

### 2. Task-Relevant

Directly measures whether generated data would be **useful for downstream prediction tasks**.

### 3. Detects Spurious Patterns

Generated data that matches statistics but violates biology will score poorly.

**Example**: A model that generates gene expression with correct marginals and correlations but violates pathway logic will produce inconsistent predictions.

### 4. Complementary to Other Metrics

Works alongside statistical metrics (FID, correlations) and biological metrics (pathway enrichment).

### 5. Interpretable

Easy to explain: "Do generated samples produce the same predictions as real samples?"

---

## Limitations and Caveats

### 1. Predictor Quality Dependency

**Issue**: Consistency is only meaningful if the predictor is accurate.

**Solutions**:

- Validate predictor performance on held-out real data first
- Use multiple predictors (ensemble approach)
- Report predictor accuracy alongside consistency scores

**Example**:
```python
# Validate predictor first
predictor_r2 = evaluate_predictor(predictor, X_test, Y_test)
if predictor_r2 < 0.7:
    print("Warning: Predictor has low accuracy, consistency may be unreliable")
```

### 2. Circular Reasoning Risk

**Issue**: If the generative model was explicitly trained to match the predictor, high consistency is expected but not informative.

**Solutions**:

- Use predictors trained independently
- Use held-out predictors not seen during generative model training
- Test on orthogonal prediction tasks

### 3. Novel Biology Detection

**Issue**: Generated samples representing **valid but novel** biological states may score poorly.

**Example**: A generative model discovers a new cell state that the predictor hasn't seen.

**Solutions**:

- Distinguish "inconsistent with predictor" from "biologically invalid"
- Investigate low-consistency samples manually
- Use multiple predictors covering different aspects of biology

### 4. Mode Collapse Masking

**Issue**: High consistency could result from generating only "safe" samples the predictor handles well.

**Solutions**:

- Also measure diversity (epiplexity, coverage metrics)
- Check prediction variance on generated samples
- Visualize generated sample distribution

**Example**:
```python
# Check if generated samples are diverse
diversity_score = compute_diversity(X_gen)
if consistency['wasserstein'] < 0.1 and diversity_score < 0.5:
    print("Warning: High consistency but low diversity - possible mode collapse")
```

### 5. Conditional Distribution Mismatch

**Issue**: Predictor may have learned $p(Y|X)$ under specific conditions not present in generated data.

**Solutions**:

- Match conditions between training and generation
- Use conditional generation with explicit condition control
- Stratify analysis by condition

---

## Best Practices

### 1. Multi-Metric Framework

**Never use prediction consistency alone**. Combine with:

| Metric Type | Examples | Purpose |
|-------------|----------|---------|
| Statistical | FID, correlations, marginals | Distribution matching |
| Biological | Pathway enrichment, markers | Biological structure |
| Prediction | Consistency (this document) | Learned relationships |
| Epiplexity | Loss curve area | Learnable structure |
| Downstream | Task performance | Practical utility |

### 2. Validate Predictors First

```python
# Step 1: Validate predictor
predictor_metrics = evaluate_predictor(predictor, X_val, Y_val)
print(f"Predictor R²: {predictor_metrics['r2']:.3f}")
print(f"Predictor MAE: {predictor_metrics['mae']:.3f}")

# Step 2: Only proceed if predictor is reliable
if predictor_metrics['r2'] > 0.7:
    consistency = distribution_consistency(predictor, X_real, X_gen)
else:
    print("Predictor not reliable enough for consistency evaluation")
```

### 3. Use Multiple Predictors

```python
# Train multiple predictors with different architectures
predictors = [
    train_linear_predictor(X_train, Y_train),
    train_rf_predictor(X_train, Y_train),
    train_nn_predictor(X_train, Y_train)
]

# Compute ensemble consistency
ensemble_metrics = ensemble_consistency(predictors, X_real, X_gen)
print(f"Mean consistency: {ensemble_metrics['mean_consistency']:.4f}")
print(f"Std consistency: {ensemble_metrics['std_consistency']:.4f}")
```

### 4. Test Multiple Prediction Tasks

```python
# Different prediction tasks
tasks = {
    'drug_response': drug_response_predictor,
    'cell_type': cell_type_classifier,
    'pathway_activity': pathway_predictor,
    'perturbation_effect': perturbation_predictor
}

# Evaluate consistency for each task
for task_name, predictor in tasks.items():
    consistency = distribution_consistency(predictor, X_real, X_gen)
    print(f"{task_name}: {consistency['wasserstein']:.4f}")
```

### 5. Stratify by Conditions

For conditional generation:

```python
# Evaluate consistency separately for each condition
conditions = ['cell_type_A', 'cell_type_B', 'cell_type_C']

for condition in conditions:
    X_real_cond = filter_by_condition(X_real, condition)
    X_gen_cond = gen_model.sample(condition=condition, n_samples=1000)
    
    consistency = distribution_consistency(
        predictor, X_real_cond, X_gen_cond
    )
    print(f"{condition}: {consistency['wasserstein']:.4f}")
```

### 6. Report Comprehensive Metrics

**Minimum reporting**:

- Predictor performance (R², accuracy, etc.)
- Consistency metric values
- Number of samples used
- Conditions tested
- Multiple predictors if available

**Example report**:
```
Prediction Consistency Evaluation
==================================
Predictor: GEM-1 (R² = 0.82 on validation)
Real samples: 10,000
Generated samples: 10,000

Consistency Metrics:
- Wasserstein distance: 0.08 ± 0.01 (3 runs)
- KS p-value: 0.23
- Mean difference: 0.03
- Variance ratio: 0.98

Interpretation: High consistency - generated samples produce
predictions similar to real data.
```

---

## Integration with Other Evaluation Methods

### Combining with Epiplexity

**Complementary insights**:

- **Epiplexity**: Does generated data teach learnable structure?
- **Prediction consistency**: Does it obey learned relationships?

**Joint interpretation**:

| Epiplexity | Consistency | Interpretation |
|------------|-------------|----------------|
| High | High | Excellent - teaches structure and obeys biology |
| High | Low | Novel patterns - investigate further |
| Low | High | Mode collapse - safe but limited samples |
| Low | Low | Poor quality - noise without structure |

### Combining with FID

**Complementary insights**:

- **FID**: Statistical distribution matching
- **Prediction consistency**: Biological relationship matching

**Example workflow**:
```python
# Compute both metrics
fid_score = compute_fid(X_real, X_gen)
consistency = distribution_consistency(predictor, X_real, X_gen)

# Joint evaluation
if fid_score < 20 and consistency['wasserstein'] < 0.1:
    print("Excellent: Good statistical and biological quality")
elif fid_score < 20 and consistency['wasserstein'] > 0.2:
    print("Warning: Good statistics but poor biological consistency")
elif fid_score > 50 and consistency['wasserstein'] < 0.1:
    print("Unusual: Poor statistics but good biological consistency")
```

---

## Case Study: Comparing Generative Models

### Scenario

Compare three generative models for gene expression:
1. VAE
2. Flow Matching
3. Diffusion Model

### Evaluation Protocol

```python
# Train all models on same data
models = {
    'VAE': train_vae(X_train),
    'Flow': train_flow_matching(X_train),
    'Diffusion': train_diffusion(X_train)
}

# Predictors
predictors = {
    'GEM-1': gem1_predictor,
    'Cell Type': cell_type_classifier,
    'Pathway': pathway_predictor
}

# Evaluate each model
results = {}
for model_name, model in models.items():
    X_gen = model.sample(n_samples=10000)
    
    results[model_name] = {}
    for pred_name, predictor in predictors.items():
        consistency = distribution_consistency(
            predictor, X_real, X_gen
        )
        results[model_name][pred_name] = consistency['wasserstein']

# Display results
import pandas as pd
df = pd.DataFrame(results).T
print(df)
```

**Example output**:
```
           GEM-1  Cell Type  Pathway
VAE         0.15       0.12     0.18
Flow        0.09       0.10     0.11
Diffusion   0.07       0.08     0.09
```

**Interpretation**: Diffusion model shows best consistency across all prediction tasks.

---

## Summary

### Key Takeaways

1. **Prediction consistency tests biological validity** through learned relationships
2. **Complements statistical metrics** like FID and correlations
3. **Requires validated predictors** - predictor quality matters
4. **Use in multi-metric framework** - never rely on single metric
5. **Interpretable and actionable** - directly measures downstream utility

### When to Use Prediction Consistency

✅ **Use when**:
- Have reliable supervised predictors
- Want to test biological validity
- Need task-relevant evaluation
- Comparing generative models for specific applications

⚠️ **Be cautious when**:
- Predictors have low accuracy
- Generative model was trained to match predictor
- Exploring novel biological states
- Limited validation data

### Recommended Workflow

1. **Validate predictors** on held-out real data
2. **Compute consistency** on generated samples
3. **Compare with other metrics** (FID, epiplexity, etc.)
4. **Investigate discrepancies** between metrics
5. **Report comprehensive results** with all metrics

---

## Related Documents

- [Evaluating Generative Models](00_evaluating_generative_models.md) — Comprehensive evaluation guide
- [FID: Fréchet Inception Distance](fid_frechet_inception_distance.md) — Statistical distribution metric
- [Epiplexity: From Entropy to Epiplexity](epiplexity/01_from_entropy_to_epiplexity.md) — Learnable structure metric
- [How Generative Models Enhance Supervised Methods](generative_models_enhance_supervised.md) — Synergy between generative and supervised learning

---

## References

### Prediction Consistency in Practice
1. **Lotfollahi, M., et al. (2023)**. Predicting cellular responses to perturbations with deep generative models. *Nature Methods*.
2. **Bunne, C., et al. (2023)**. Learning Single-Cell Perturbation Responses using Neural Optimal Transport. *Nature Methods*.

### Evaluation Frameworks
3. **Xu, Q., et al. (2018)**. An empirical study on evaluation metrics of generative adversarial networks. *arXiv*.
4. **Borji, A. (2019)**. Pros and cons of GAN evaluation measures. *Computer Vision and Image Understanding*.

### Biological Validation
5. **Eraslan, G., et al. (2019)**. Single-cell RNA-seq denoising using a deep count autoencoder. *Nature Communications*.
6. **Lopez, R., et al. (2018)**. Deep generative modeling for single-cell transcriptomics. *Nature Methods*.
