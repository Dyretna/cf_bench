# Stopping Threshold in DiCE - Implementation Guide

This document provides practical implementation examples and code snippets for using the `stopping_threshold` parameter in DiCE effectively.

---

## 1. Basic Usage

### Setting Stopping Threshold in DiCE

```python
import dice_ml
from dice_ml import Dice

# Load your data and model
d = dice_ml.Data(dataframe=df,
                 continuous_features=['bmi'],
                 outcome_name='hltprhc')

m = dice_ml.Model(model=your_model, backend='sklearn')

# Create DiCE explainer with stopping threshold
exp = Dice(d, m, method='genetic')

# Generate counterfactuals with specific threshold
cf = exp.generate_counterfactuals(
    query_instances=test_instance,
    total_CFs=10,
    desired_class=0,  # Want to flip to low-risk
    stopping_threshold=0.9,  # Target probability < 0.1
    posthoc_sparsity_param=0.1,
    verbose=False
)
```

### Understanding What Happens

With `stopping_threshold=0.9`:
- DiCE searches for instances where `P(class=1) < 1 - 0.9 = 0.1`
- Only counterfactuals meeting this criterion are considered "valid"
- Search continues until `total_CFs` valid ones are found or `maxiterations` is reached

---

## 2. Threshold Values for Different Scenarios

### Example 1: Ambitious Medical Intervention (Recommended)

**Goal:** Reduce cardiovascular risk to "low risk" category (<10%)

```python
# Configuration for meaningful health improvement
cf = exp.generate_counterfactuals(
    query_instances=patient_data,
    total_CFs=10,
    desired_class=0,
    stopping_threshold=0.9,  # Target: <10% risk
    maxiterations=1000,
    verbose=True
)

# Check results
if cf.cf_examples_list[0].final_cfs_df is not None:
    print("Success! Generated counterfactuals with <10% risk")
    print(cf.cf_examples_list[0].final_cfs_df)
else:
    print("Failed - could not reach <10% risk target")
```

**When to use:** Well-calibrated models, clinical applications requiring significant risk reduction

---

### Example 2: Moderate Target

**Goal:** Reduce risk to medium-low category (<30%)

```python
cf = exp.generate_counterfactuals(
    query_instances=patient_data,
    total_CFs=10,
    desired_class=0,
    stopping_threshold=0.7,  # Target: <30% risk
    maxiterations=1000,
    verbose=False
)
```

**When to use:** Moderately inflated probabilities, intermediate goals, exploratory analysis

---

### Example 3: Conservative Baseline

**Goal:** Just flip the prediction (from >50% to <50%)

```python
cf = exp.generate_counterfactuals(
    query_instances=patient_data,
    total_CFs=10,
    desired_class=0,
    stopping_threshold=0.5,  # Target: <50% risk (just flip)
    maxiterations=1000,
    verbose=False
)
```

**When to use:** Class-weighted models, proof of concept, ensuring high success rates

---

## 3. Testing Multiple Thresholds

### Systematic Threshold Comparison

```python
import pandas as pd

def test_thresholds(exp, test_data, thresholds=[0.1, 0.5, 0.7, 0.9]):
    """
    Test CF generation with multiple stopping thresholds.

    Parameters
    ----------
    exp : Dice
        DiCE explainer object
    test_data : pd.DataFrame
        Test instances to generate CFs for
    thresholds : list
        List of stopping_threshold values to test

    Returns
    -------
    pd.DataFrame
        Results summary for each threshold
    """
    results = []

    for thresh in thresholds:
        print(f"\nTesting threshold: {thresh} (target prob < {1-thresh:.2f})")

        n_success = 0
        n_valid = 0
        n_total_generated = 0

        for idx in test_data.index:
            query = test_data.loc[[idx]]

            try:
                cf = exp.generate_counterfactuals(
                    query_instances=query,
                    total_CFs=10,
                    desired_class=0,
                    stopping_threshold=thresh,
                    maxiterations=1000,
                    verbose=False
                )

                cf_df = cf.cf_examples_list[0].final_cfs_df

                if cf_df is not None and len(cf_df) > 0:
                    n_success += 1
                    n_valid += len(cf_df)
                    n_total_generated += 10
                else:
                    n_total_generated += 10

            except Exception as e:
                print(f"Error for instance {idx}: {e}")
                n_total_generated += 10

        results.append({
            'threshold': thresh,
            'target_prob': 1 - thresh,
            'query_success_rate': n_success / len(test_data),
            'valid_cfs': n_valid,
            'total_generated': n_total_generated,
            'validity_rate': n_valid / n_total_generated if n_total_generated > 0 else 0
        })

    return pd.DataFrame(results)

# Usage
results_df = test_thresholds(exp, test_data)
print("\n" + "="*70)
print("THRESHOLD COMPARISON RESULTS")
print("="*70)
print(results_df.to_string(index=False))
```

**Example output:**
```
======================================================================
THRESHOLD COMPARISON RESULTS
======================================================================
threshold  target_prob  query_success_rate  valid_cfs  total_generated  validity_rate
      0.1         0.90            1.000000         45               90       0.500000
      0.5         0.50            0.888889         38               90       0.422222
      0.7         0.30            0.888889         35               90       0.388889
      0.9         0.10            1.000000         35               90       0.388889
```

---

## 4. Checking Model Compatibility

### Diagnose Probability Distribution

Before choosing a threshold, check your model's probability distribution:

```python
import numpy as np
import matplotlib.pyplot as plt

def check_model_probabilities(model, X_test, y_test):
    """
    Analyze model probability distribution to guide threshold selection.

    Parameters
    ----------
    model : sklearn model
        Trained classifier
    X_test : array-like
        Test features
    y_test : array-like
        Test labels

    Returns
    -------
    dict
        Statistics about probability distribution
    """
    # Get predictions
    probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1

    # Calculate statistics
    stats = {
        'mean_prob': np.mean(probs),
        'median_prob': np.median(probs),
        'min_prob': np.min(probs),
        'max_prob': np.max(probs),
        'std_prob': np.std(probs),
        'below_0.1': np.sum(probs < 0.1) / len(probs),
        'below_0.3': np.sum(probs < 0.3) / len(probs),
        'below_0.5': np.sum(probs < 0.5) / len(probs)
    }

    # Print report
    print("="*60)
    print("MODEL PROBABILITY DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"\nProbability statistics:")
    print(f"  Mean:   {stats['mean_prob']:.3f}")
    print(f"  Median: {stats['median_prob']:.3f}")
    print(f"  Range:  [{stats['min_prob']:.3f}, {stats['max_prob']:.3f}]")
    print(f"  Std:    {stats['std_prob']:.3f}")

    print(f"\nDistribution:")
    print(f"  {stats['below_0.1']*100:.1f}% of instances have P(class=1) < 0.1")
    print(f"  {stats['below_0.3']*100:.1f}% of instances have P(class=1) < 0.3")
    print(f"  {stats['below_0.5']*100:.1f}% of instances have P(class=1) < 0.5")

    # Recommendation
    print(f"\nRecommended stopping_threshold:")
    if stats['mean_prob'] < 0.15:
        print("  ✓ Use 0.9 (model well-calibrated, low probabilities)")
    elif stats['mean_prob'] < 0.35:
        print("  ⚠ Use 0.7-0.8 (moderate probabilities)")
    else:
        print("  ✗ Use 0.5 or fix model (inflated probabilities)")
        print("    Model may have class weighting or probability calibration issues")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0.1, color='g', linestyle='--', linewidth=2, label='Threshold 0.9 target')
    plt.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='Threshold 0.7 target')
    plt.axvline(0.5, color='r', linestyle='--', linewidth=2, label='Threshold 0.5 target')
    plt.xlabel('P(class=1)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Model Probability Distribution', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_probability_distribution.png', dpi=300)
    print(f"\nPlot saved to: model_probability_distribution.png")

    return stats

# Usage
stats = check_model_probabilities(model, X_test, y_test)
```

**Example output for well-calibrated model:**
```
============================================================
MODEL PROBABILITY DISTRIBUTION ANALYSIS
============================================================

Probability statistics:
  Mean:   0.110
  Median: 0.089
  Range:  [0.013, 0.327]
  Std:    0.078

Distribution:
  65.3% of instances have P(class=1) < 0.1
  89.2% of instances have P(class=1) < 0.3
  98.6% of instances have P(class=1) < 0.5

Recommended stopping_threshold:
  ✓ Use 0.9 (model well-calibrated, low probabilities)
```

**Example output for problematic model:**
```
============================================================
MODEL PROBABILITY DISTRIBUTION ANALYSIS
============================================================

Probability statistics:
  Mean:   0.438
  Median: 0.452
  Range:  [0.254, 0.606]
  Std:    0.112

Distribution:
  0.0% of instances have P(class=1) < 0.1
  8.3% of instances have P(class=1) < 0.3
  43.1% of instances have P(class=1) < 0.5

Recommended stopping_threshold:
  ✗ Use 0.5 or fix model (inflated probabilities)
    Model may have class weighting or probability calibration issues
```

---

## 5. Adaptive Threshold Selection

### Auto-adjust Based on Success Rate

```python
def generate_cfs_adaptive_threshold(exp, query_instances,
                                     initial_threshold=0.9,
                                     min_success_rate=0.8):
    """
    Generate CFs with adaptive threshold adjustment.
    Starts with ambitious threshold and reduces if success rate is too low.

    Parameters
    ----------
    exp : Dice
        DiCE explainer
    query_instances : pd.DataFrame
        Instances to generate CFs for
    initial_threshold : float
        Starting threshold (default: 0.9)
    min_success_rate : float
        Minimum acceptable success rate (default: 0.8)

    Returns
    -------
    dict
        Results with final threshold and CFs
    """
    thresholds = [0.9, 0.7, 0.5, 0.3]
    thresholds = [t for t in thresholds if t <= initial_threshold]

    for thresh in thresholds:
        print(f"\nTrying threshold: {thresh} (target prob < {1-thresh:.2f})")

        success_count = 0
        all_cfs = []

        for idx in query_instances.index:
            query = query_instances.loc[[idx]]

            try:
                cf = exp.generate_counterfactuals(
                    query_instances=query,
                    total_CFs=10,
                    desired_class=0,
                    stopping_threshold=thresh,
                    maxiterations=1000,
                    verbose=False
                )

                cf_df = cf.cf_examples_list[0].final_cfs_df

                if cf_df is not None and len(cf_df) > 0:
                    success_count += 1
                    all_cfs.append(cf)
                else:
                    all_cfs.append(None)

            except Exception as e:
                print(f"Error: {e}")
                all_cfs.append(None)

        success_rate = success_count / len(query_instances)
        print(f"Success rate: {success_rate*100:.1f}%")

        if success_rate >= min_success_rate:
            print(f"✓ Acceptable success rate achieved with threshold {thresh}")
            return {
                'threshold': thresh,
                'success_rate': success_rate,
                'counterfactuals': all_cfs
            }

    print(f"⚠ Could not achieve {min_success_rate*100:.0f}% success rate")
    return {
        'threshold': thresholds[-1],
        'success_rate': success_rate,
        'counterfactuals': all_cfs
    }

# Usage
result = generate_cfs_adaptive_threshold(exp, test_data,
                                         initial_threshold=0.9,
                                         min_success_rate=0.8)

print(f"\nFinal configuration:")
print(f"  Threshold: {result['threshold']}")
print(f"  Success rate: {result['success_rate']*100:.1f}%")
```

---

## 6. Validating Generated CFs

### Check if CFs Meet Threshold

```python
def validate_cf_probabilities(model, original_instance, cf_df,
                              stopping_threshold):
    """
    Validate that generated CFs actually meet the stopping threshold.

    Parameters
    ----------
    model : sklearn model
        The predictive model
    original_instance : pd.DataFrame
        Original instance
    cf_df : pd.DataFrame
        Generated counterfactuals
    stopping_threshold : float
        The threshold used in generation

    Returns
    -------
    pd.DataFrame
        Validation results
    """
    target_prob = 1 - stopping_threshold

    # Get original probability
    orig_prob = model.predict_proba(original_instance)[:, 1][0]

    # Get CF probabilities
    cf_probs = model.predict_proba(cf_df)[:, 1]

    # Check validity
    valid_mask = cf_probs < target_prob

    # Create results
    results = pd.DataFrame({
        'cf_id': range(len(cf_df)),
        'predicted_prob': cf_probs,
        'target_prob': target_prob,
        'meets_threshold': valid_mask,
        'prob_reduction': orig_prob - cf_probs
    })

    print("="*70)
    print("COUNTERFACTUAL VALIDATION")
    print("="*70)
    print(f"\nOriginal probability: {orig_prob:.4f}")
    print(f"Target probability: < {target_prob:.4f}")
    print(f"Stopping threshold: {stopping_threshold}")
    print(f"\n{valid_mask.sum()}/{len(cf_df)} CFs meet threshold")
    print(f"\nProbability statistics:")
    print(f"  Min CF prob: {cf_probs.min():.4f}")
    print(f"  Max CF prob: {cf_probs.max():.4f}")
    print(f"  Mean CF prob: {cf_probs.mean():.4f}")
    print(f"  Mean reduction: {(orig_prob - cf_probs).mean():.4f}")

    return results

# Usage
validation = validate_cf_probabilities(
    model=your_model,
    original_instance=test_instance,
    cf_df=generated_cfs,
    stopping_threshold=0.9
)

print("\n" + validation.to_string(index=False))
```

---

## 7. Configuration Files

### YAML Configuration Example

```yaml
# config_ambitious_threshold.yaml
model_path: "models/xgboost_optimized.pkl"
data_path: "data/cfcheck.csv"

counterfactual_generation:
  total_CFs: 10
  stopping_threshold: 0.9  # Target probability < 0.1 (10% risk)
  desired_class: 0
  method: "genetic"

  # Other parameters
  posthoc_sparsity_param: 0.1
  posthoc_sparsity_algorithm: "linear"
  maxiterations: 1000
  verbose: false

  # Feature configuration
  continuous_features: ["bmi"]
  ordinal_features: ["etfruit", "eatveg", "cgtsmok", "alcfreq",
                     "slprl", "paccnois", "dosprt"]
  features_to_vary: ["etfruit", "eatveg", "cgtsmok", "alcfreq",
                     "slprl", "paccnois", "bmi", "dosprt"]

  # Constraints
  permitted_range:
    bmi: [16.0, 45.0]
    etfruit: [1, 7]
    # ... other ranges
```

### Loading Configuration

```python
import yaml

def load_cf_config(config_path):
    """Load CF generation config from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_cfs_from_config(exp, query_instances, config):
    """Generate CFs using configuration."""
    cf_config = config['counterfactual_generation']

    cf = exp.generate_counterfactuals(
        query_instances=query_instances,
        total_CFs=cf_config['total_CFs'],
        desired_class=cf_config['desired_class'],
        stopping_threshold=cf_config['stopping_threshold'],
        posthoc_sparsity_param=cf_config['posthoc_sparsity_param'],
        posthoc_sparsity_algorithm=cf_config['posthoc_sparsity_algorithm'],
        maxiterations=cf_config['maxiterations'],
        verbose=cf_config['verbose']
    )

    return cf

# Usage
config = load_cf_config('configs/ambitious_threshold.yaml')
cfs = generate_cfs_from_config(exp, test_data, config)
```

---

## 8. Troubleshooting Common Issues

### Issue 1: No valid CFs generated

```python
def debug_cf_failure(exp, query_instance, stopping_threshold=0.9):
    """
    Debug why CF generation fails for an instance.

    Parameters
    ----------
    exp : Dice
        DiCE explainer
    query_instance : pd.DataFrame
        The problematic instance
    stopping_threshold : float
        The threshold being used
    """
    print("="*70)
    print("DEBUGGING CF GENERATION FAILURE")
    print("="*70)

    # Get model prediction
    model = exp.model.model
    orig_prob = model.predict_proba(query_instance)[:, 1][0]
    target_prob = 1 - stopping_threshold
    distance_to_target = orig_prob - target_prob

    print(f"\n1. Probability Analysis:")
    print(f"   Original probability: {orig_prob:.4f}")
    print(f"   Target probability: < {target_prob:.4f}")
    print(f"   Distance to target: {distance_to_target:.4f}")

    if orig_prob < target_prob:
        print("   ✓ Already meets target! (shouldn't fail)")
    elif distance_to_target > 0.3:
        print(f"   ✗ LARGE distance to target ({distance_to_target:.2f})")
        print("   → Consider lowering threshold")
    else:
        print(f"   ⚠ Moderate distance ({distance_to_target:.2f})")
        print("   → May be achievable with more iterations")

    # Try with more iterations
    print(f"\n2. Testing with increased iterations:")
    try:
        cf = exp.generate_counterfactuals(
            query_instances=query_instance,
            total_CFs=10,
            desired_class=0,
            stopping_threshold=stopping_threshold,
            maxiterations=5000,  # Increase iterations
            verbose=True
        )

        if cf.cf_examples_list[0].final_cfs_df is not None:
            print("   ✓ Success with more iterations!")
        else:
            print("   ✗ Still failed with 5000 iterations")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Try with lower threshold
    print(f"\n3. Testing with lower threshold:")
    lower_thresholds = [0.7, 0.5, 0.3]

    for thresh in lower_thresholds:
        if thresh >= stopping_threshold:
            continue

        try:
            cf = exp.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=10,
                desired_class=0,
                stopping_threshold=thresh,
                maxiterations=1000,
                verbose=False
            )

            if cf.cf_examples_list[0].final_cfs_df is not None:
                n_valid = len(cf.cf_examples_list[0].final_cfs_df)
                print(f"   ✓ Threshold {thresh} works! ({n_valid} valid CFs)")
                break
            else:
                print(f"   ✗ Threshold {thresh} failed")

        except Exception as e:
            print(f"   ✗ Threshold {thresh} error: {e}")

    print("\n" + "="*70)

# Usage
debug_cf_failure(exp, problematic_instance, stopping_threshold=0.9)
```

---

## 9. Best Practices Implementation

### Production-Ready CF Generation

```python
class CFGenerator:
    """
    Production-ready CF generator with threshold management.
    """

    def __init__(self, exp, default_threshold=0.9,
                 min_success_rate=0.8):
        """
        Initialize CF generator.

        Parameters
        ----------
        exp : Dice
            DiCE explainer object
        default_threshold : float
            Default stopping threshold
        min_success_rate : float
            Minimum acceptable success rate
        """
        self.exp = exp
        self.default_threshold = default_threshold
        self.min_success_rate = min_success_rate
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'threshold_used': default_threshold
        }

    def generate(self, query_instances, total_CFs=10,
                 adaptive=True, **kwargs):
        """
        Generate counterfactuals with optional adaptive threshold.

        Parameters
        ----------
        query_instances : pd.DataFrame
            Instances to generate CFs for
        total_CFs : int
            Number of CFs to generate per instance
        adaptive : bool
            Whether to use adaptive threshold adjustment
        **kwargs : dict
            Additional arguments for generate_counterfactuals

        Returns
        -------
        list
            List of CF objects
        """
        if adaptive:
            return self._generate_adaptive(query_instances, total_CFs,
                                          **kwargs)
        else:
            return self._generate_fixed(query_instances, total_CFs,
                                       **kwargs)

    def _generate_fixed(self, query_instances, total_CFs, **kwargs):
        """Generate with fixed threshold."""
        threshold = kwargs.pop('stopping_threshold', self.default_threshold)

        results = []
        for idx in query_instances.index:
            query = query_instances.loc[[idx]]

            cf = self.exp.generate_counterfactuals(
                query_instances=query,
                total_CFs=total_CFs,
                stopping_threshold=threshold,
                **kwargs
            )

            results.append(cf)
            self.stats['total_queries'] += 1

            if cf.cf_examples_list[0].final_cfs_df is not None:
                self.stats['successful_queries'] += 1

        return results

    def _generate_adaptive(self, query_instances, total_CFs, **kwargs):
        """Generate with adaptive threshold adjustment."""
        thresholds = [0.9, 0.7, 0.5, 0.3]

        for thresh in thresholds:
            results = self._generate_fixed(
                query_instances,
                total_CFs,
                stopping_threshold=thresh,
                **kwargs
            )

            success_rate = (self.stats['successful_queries'] /
                           self.stats['total_queries'])

            if success_rate >= self.min_success_rate:
                self.stats['threshold_used'] = thresh
                return results

        # Use last threshold if nothing worked
        self.stats['threshold_used'] = thresholds[-1]
        return results

    def get_stats(self):
        """Get generation statistics."""
        if self.stats['total_queries'] == 0:
            return self.stats

        success_rate = (self.stats['successful_queries'] /
                       self.stats['total_queries'])

        return {
            **self.stats,
            'success_rate': success_rate
        }

# Usage
generator = CFGenerator(exp, default_threshold=0.9, min_success_rate=0.8)

# Generate with adaptive threshold
cfs = generator.generate(
    test_data,
    total_CFs=10,
    desired_class=0,
    adaptive=True,
    maxiterations=1000,
    verbose=False
)

# Check stats
stats = generator.get_stats()
print(f"\nGeneration statistics:")
print(f"  Total queries: {stats['total_queries']}")
print(f"  Successful: {stats['successful_queries']}")
print(f"  Success rate: {stats['success_rate']*100:.1f}%")
print(f"  Threshold used: {stats['threshold_used']}")
```

---

## 10. Summary Checklist

**Before choosing a stopping threshold:**

- [ ] Check model probability distribution
- [ ] Verify model has no class weighting (or calibrate probabilities)
- [ ] Understand domain requirements (how much risk reduction is meaningful?)
- [ ] Test threshold on small sample first

**When implementing:**

- [ ] Start with threshold 0.9 for well-calibrated models
- [ ] Use adaptive threshold adjustment for robustness
- [ ] Monitor query success rate (aim for >80%)
- [ ] Validate generated CFs actually meet threshold
- [ ] Document threshold choice and rationale
- [ ] Save configuration for reproducibility

**If CFs fail to generate:**

- [ ] Check model probabilities (may be inflated)
- [ ] Try lower threshold (0.7, then 0.5)
- [ ] Increase maxiterations
- [ ] Check other constraints (permitted_range, features_to_vary)
- [ ] Consider model retraining without class weighting

---

**Document Version**: 1.0
**Date**: May 7, 2026
**Related**: DiCE_stopping_threshold_overview.md
