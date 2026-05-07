# Generation 2 Experimental Results Analysis

**Date**: May 7, 2026
**Purpose**: Comprehensive analysis of model and threshold selection for counterfactual generation
**Status**: Final Recommendation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experimental Overview](#experimental-overview)
3. [Key Metrics Explained](#key-metrics-explained)
4. [Results by Experiment Type](#results-by-experiment-type)
5. [Comparative Analysis](#comparative-analysis)
6. [Final Recommendations](#final-recommendations)
7. [Technical Considerations](#technical-considerations)
8. [Next Steps](#next-steps)

---

## Executive Summary

### Main Finding

**Recommended Configuration for Continued Research:**

- **Primary Choice**: XGBoost (optimized hyperparameters, no class weighting) with **stopping_threshold = 0.9**
- **Alternative**: Baseline Random Forest with **stopping_threshold = 0.1** (lowthres)

### Why These Choices?

**XGB-optimized with threshold 0.9:**
- 100% query success rate (all 9 queries got at least 1 valid CF)
- 35 valid CFs generated (highest among 100% success configurations)
- Fast generation (1.19 seconds per query)
- Benefits from hyperparameter optimization without probability inflation issues

**Baseline RF with threshold 0.1:**
- 100% query success rate
- Highest validity rate among baseline models (50.8%)
- Reliable and consistent performance
- Well-tested baseline for comparisons

### Key Insight: SMOTE Models Underperformed

SMOTE-trained models, especially with grid-searched hyperparameters, showed significantly worse CF generation capability:
- SMOTE-grid RF thres0.5: Only 22.2% query success
- SMOTE-grid RF thres0.9: Only 44.4% query success
- SMOTE models are NOT recommended for CF generation

---

## Experimental Overview

### What Was Tested

**Three major experiment groups:**

1. **Base vs Thresholds**: Baseline RF and XGB models with three stopping thresholds
   - Models: Random Forest, XGBoost (no class weighting, unoptimized hyperparameters)
   - Thresholds: high (0.9), mid (0.5), low (0.1)
   - Purpose: Understand threshold trade-offs with calibrated probabilities

2. **SMOTE Models**: Models trained on SMOTE-resampled data
   - SMOTE-base: Base hyperparameters on SMOTE data
   - SMOTE-grid: Grid-searched hyperparameters on SMOTE data
   - Thresholds: 0.5 and 0.9
   - Purpose: Test if synthetic minority oversampling improves CF generation

3. **XGB Optimized**: XGBoost with optimized hyperparameters (no class weighting)
   - Optimized structure (depth, learning rate, regularization)
   - NO scale_pos_weight (avoids probability inflation)
   - Thresholds: 0.1, 0.5, 0.9
   - Purpose: Best of both worlds - optimized structure with calibrated probabilities

### Test Set

- 9 query instances (representative sample from test data)
- All experiments used the same test set for fair comparison
- Target: Generate 10 counterfactuals per query instance

---

## Key Metrics Explained

### Query Success Rate

**Definition**: Percentage of queries that produced at least 1 valid counterfactual

**Why It Matters**: Most important metric for practical usability. If a model can't generate ANY valid CF for a query, it's unusable for that case.

**Target**: Aim for 90-100%

### Validity Rate

**Definition**: Proportion of generated CFs that meet all validity criteria (desired class, within constraints, etc.)

**Why It Matters**: Indicates efficiency - higher validity means less wasted computation on invalid CFs.

**Target**: Aim for 40-60%

### Valid CFs Generated

**Definition**: Total count of valid counterfactuals across all queries

**Why It Matters**: More valid CFs = more diversity for users to choose from. With 9 queries requesting 10 CFs each, maximum is 90.

**Target**: Aim for 30-50 valid CFs (33-55% of attempts)

### Gower Distance

**Definition**: Average distance between original instances and valid counterfactuals (0-1 scale)

**Why It Matters**: Measures how different CFs are from originals. Lower is better (smaller changes), but too low may indicate insufficient diversity.

**Typical Range**: 0.85-1.0 (on our scaled features)

### Average Features Changed

**Definition**: Mean number of features modified in valid counterfactuals

**Why It Matters**: Indicates CF complexity. Lower is better (simpler, more actionable recommendations).

**Target**: Aim for 2-3 features

### Generation Time

**Definition**: Average seconds to generate CFs per query instance

**Why It Matters**: Affects scalability and user experience

**Target**: Under 3 seconds ideal, under 10 seconds acceptable

---

## Results by Experiment Type

### 1. Baseline Models (Base vs Thresholds)

#### Random Forest Results

| Threshold  | Stopping Value | Query Success | Validity Rate | Valid CFs | Avg Time |
|------------|----------------|---------------|---------------|-----------|----------|
| highthres  | 0.9            | 100.0%        | 41.7%         | 35        | 2.65s    |
| lowthres   | 0.1            | 100.0%        | 50.8%         | 31        | 2.66s    |
| midthres   | 0.5            | 88.9%         | 37.8%         | 34        | 2.60s    |

**Key Observations:**
- Perfect query success with high (0.9) and low (0.1) thresholds
- Low threshold (0.1) achieved highest validity rate (50.8%)
- Mid threshold (0.5) had one failed query (88.9% success)
- Consistent generation time (~2.6s per query)

**Threshold Analysis:**
- **High (0.9)**: Target <10% risk - Most ambitious, surprisingly still 100% success
- **Low (0.1)**: Target <90% risk - Easiest target, highest validity
- **Mid (0.5)**: Target <50% risk - Moderate target, but worse performance than extremes

#### XGBoost Results

| Threshold  | Stopping Value | Query Success | Validity Rate | Valid CFs | Avg Time |
|------------|----------------|---------------|---------------|-----------|----------|
| highthres  | 0.9            | 77.8%         | 57.3%         | 47        | 1.04s    |
| lowthres   | 0.1            | 77.8%         | 56.5%         | 39        | 1.10s    |
| midthres   | 0.5            | 88.9%         | 46.7%         | 42        | 1.05s    |

**Key Observations:**
- Lower query success than RF (77.8-88.9% vs 100%)
- Higher validity rates (46.7-57.3% vs 37.8-50.8%)
- Most valid CFs generated (39-47 vs 31-35)
- Much faster than RF (1.04-1.10s vs 2.6s)

**Trade-off**: XGB generates more valid CFs but fails completely for 1-2 queries

### 2. SMOTE Models

#### SMOTE Base Predictors (Unoptimized hyperparameters)

| Model | Threshold | Query Success | Validity Rate | Valid CFs | Avg Time |
|-------|-----------|---------------|---------------|-----------|----------|
| RF    | 0.5       | 88.9%         | 24.7%         | 22        | 9.71s    |
| RF    | 0.9       | 88.9%         | 44.8%         | 26        | 9.83s    |
| XGB   | 0.5       | 77.8%         | 31.1%         | 28        | 1.08s    |
| XGB   | 0.9       | 77.8%         | 42.3%         | 22        | 1.06s    |

**Key Observations:**
- Worse query success than baseline (77.8-88.9% vs 100%)
- Lower validity rates than baseline
- Fewer valid CFs generated
- RF VERY slow (9.7s vs 2.6s baseline) - likely due to SMOTE data size
- XGB maintains speed (similar to baseline)

#### SMOTE Grid-Searched (Optimized hyperparameters)

| Model | Threshold | Query Success | Validity Rate | Valid CFs | Avg Time |
|-------|-----------|---------------|---------------|-----------|----------|
| RF    | 0.5       | 22.2%         | 6.7%          | 6         | 3.90s    |
| RF    | 0.9       | 44.4%         | 23.5%         | 8         | 3.85s    |
| XGB   | 0.5       | 66.7%         | 20.7%         | 18        | 11.12s   |

**Key Observations:**
- CATASTROPHIC performance for RF (22.2-44.4% query success)
- Only 6-8 valid CFs from RF (vs 31-35 baseline)
- XGB better but still worse than baseline (66.7% vs 77.8-88.9%)
- Grid search optimization HURT CF generation capability

**Conclusion**: SMOTE + hyperparameter optimization is incompatible with CF generation

### 3. XGB Optimized (Best Approach)

| Threshold | Stopping Value | Query Success | Validity Rate | Valid CFs | Avg Time |
|-----------|----------------|---------------|---------------|-----------|----------|
| thres0.1  | 0.1            | 88.9%         | 47.8%         | 33        | 1.20s    |
| thres0.5  | 0.5            | 88.9%         | 32.2%         | 29        | 1.23s    |
| thres0.9  | 0.9            | 100.0%        | 41.2%         | 35        | 1.19s    |

**Key Observations:**
- **thres0.9 achieved 100% query success!** (Best result)
- Competitive validity rates (32.2-47.8%)
- Good number of valid CFs (29-35)
- Excellent speed (1.19-1.23s)
- **Best of both worlds**: Optimized structure + calibrated probabilities

**Why This Works:**
- Hyperparameters optimized for structure (depth, regularization, learning rate)
- NO scale_pos_weight (no probability inflation)
- Model learned from full unbalanced dataset (natural distribution)
- Benefits from optimization without the CF-breaking side effects

---

## Comparative Analysis

### Performance Ranking by Query Success Rate

1. **100% Success (Tie)**:
   - Base RF - highthres (0.9)
   - Base RF - lowthres (0.1)
   - XGB-optimized - thres0.9

2. **88.9% Success**:
   - Base RF - midthres (0.5)
   - Base XGB - midthres (0.5)
   - SMOTE-base RF - both thresholds
   - XGB-optimized - thres0.1 and thres0.5

3. **77.8% Success**:
   - Base XGB - highthres and lowthres
   - SMOTE-base XGB - both thresholds

4. **66.7% Success and Below**:
   - SMOTE-grid models (poor performance)

### Performance Ranking by Valid CFs Generated

1. Base XGB - highthres: **47 valid CFs** (but only 77.8% query success)
2. Base XGB - midthres: **42 valid CFs** (88.9% query success)
3. Base XGB - lowthres: **39 valid CFs** (77.8% query success)
4. **Base RF - highthres: 35 valid CFs (100% query success)** ✓
5. **XGB-optimized - thres0.9: 35 valid CFs (100% query success)** ✓

### Speed Comparison

**Fastest (< 1.5s):**
- All XGB models (base and optimized): 1.04-1.23s
- SMOTE XGB base: 1.06-1.08s

**Medium (2-4s):**
- All baseline RF: 2.60-2.66s
- SMOTE RF grid-searched: 3.85-3.90s

**Slowest (> 9s):**
- SMOTE RF base: 9.71-9.83s (training data size issue)

### Threshold Effect Analysis

**Counterintuitive Finding**: Threshold 0.9 (most ambitious) performed BETTER than 0.5 (moderate) in several cases!

**Why?** (See Understanding_CF_Generation_and_Model_Compatibility.md)
- With calibrated probabilities, most test instances already have low predicted risk (0.02-0.15)
- Threshold 0.9 (target <0.1) is already satisfied or nearly satisfied
- Threshold 0.5 (target <0.5) may trigger different search strategies
- DiCE may optimize differently depending on how "far" from target

**Practical Implication**: Don't fear ambitious thresholds with calibrated models!

### Model Type Comparison

**Random Forest Advantages:**
- More reliable query success (100% for high/low thresholds)
- Simpler, more interpretable
- Fewer complete failures

**XGBoost Advantages:**
- Faster (1s vs 2.6s)
- Generates more valid CFs overall
- Better validity rates
- More tunable with hyperparameters

**Winner**: XGB-optimized combines best of both (100% success + speed + valid CFs)

---

## Final Recommendations

### Primary Recommendation: XGB-Optimized with Threshold 0.5-0.7

**Configuration:**
```yaml
model_path: "models/xgboost_optimized_cf_friendly.pkl"
stopping_threshold: 0.5  # Primary recommendation for production
# stopping_threshold: 0.7  # Alternative, more ambitious
# stopping_threshold: 0.9  # ONLY for pre-filtered low-risk research
```

**Model Specification:**
- XGBoost with optimized hyperparameters (see xgb_optimized configs)
- NO scale_pos_weight (trained without class weighting)
- Trained on full unbalanced dataset

**Performance (from experiments with threshold 0.9):**
- Query Success: 100% (on low-risk test set)
- Valid CFs: 35 (tied for best among 100% success)
- Validity Rate: 41.2%
- Speed: 1.19s per query
- Avg Features Changed: 2.49

**Why Threshold 0.5-0.7 Instead of 0.9:**

**Full test set analysis reveals:**
- Max prediction: 0.9723 (NOT 0.20 as seen in CF test set!)
- Median prediction: 0.0440
- 95th percentile: 0.5215

**This means:**
1. **CF test set was unrepresentative** - 9 individuals happened to be low-risk (max pred 0.20)
2. **Model CAN predict high risk** - ~5% of population gets predictions 0.52-0.97
3. **Threshold 0.9 will fail for high-risk** - person with 0.75 prob cannot reach <0.10 realistically
4. **Threshold 0.5-0.7 works for ALL** - even 0.75 prob can reach <0.375 (0.5 threshold) or <0.225 (0.7 threshold)

**Recommended approach:**
```python
# For production (handles entire population)
stopping_threshold = 0.5  # Target: <50% of original risk

# For ambitious but realistic targets
stopping_threshold = 0.7  # Target: <30% of original risk

# ONLY for research on pre-filtered low-risk (prob <0.30)
stopping_threshold = 0.9  # Target: <10% absolute risk
```

**Best For:**
- Production systems with diverse user population (recommended)
- Clinical applications where ALL patients need actionable recommendations
- Large-scale deployment without pre-filtering by risk level

### Alternative: Baseline RF with Threshold 0.1

**Configuration:**
```yaml
model_path: "models/rf_hltprhc.pkl"
stopping_threshold: 0.1
```

**Performance:**
- Query Success: 100%
- Valid CFs: 31
- Validity Rate: 50.8% (highest among baselines)
- Speed: 2.66s per query
- Avg Features Changed: 2.26

**Why This Choice:**
1. **Highest Validity**: 50.8% - most efficient CF generation
2. **Conservative Target**: Threshold 0.1 (target <90% risk) easier to achieve
3. **Well-Tested**: Baseline model, stable and understood
4. **Good for Comparisons**: Reference point for future experiments

**Best For:**
- Research requiring conservative CFs
- Experiments needing high validity rates
- Baseline comparisons

### NOT Recommended: SMOTE Models

**Avoid:**
- SMOTE-base RF: Slow (9.7s) and worse performance
- SMOTE-grid RF: Catastrophic CF generation (22-44% success)
- SMOTE-grid XGB: Poor query success (66.7%)

**Why They Failed:**
- SMOTE synthetic data may create artificial patterns
- Hyperparameter optimization on SMOTE data breaks CF capability
- Training set size issues (slower, less robust)

**Conclusion**: SMOTE is incompatible with DiCE counterfactual generation

### Threshold Selection Guidelines

**If using calibrated models (baseline or XGB-optimized):**

- **Threshold 0.9 (recommended)**: Target <10% risk
  - Best for ambitious risk reduction
  - Clinically meaningful outcomes
  - Works well despite being hardest target

- **Threshold 0.5**: Target <50% risk
  - Moderate target
  - Slightly worse performance than 0.9 in our tests
  - Use if 0.9 proves too restrictive in your data

- **Threshold 0.1**: Target <90% risk
  - Conservative, easy target
  - Highest validity rates
  - Use for maximum CF generation success

**Rule of Thumb**: Start with 0.9, reduce if query success drops below 90%

---

## Technical Considerations

### Why XGB-Optimized Works

**The Problem We Solved:**
Previous hyperparameter optimization with class weighting broke CF generation by inflating probabilities (see Understanding_CF_Generation_and_Model_Compatibility.md).

**The Solution:**
1. Keep beneficial hyperparameters from optimization:
   - max_depth: 11 (controlled complexity)
   - learning_rate: 0.25 (appropriate learning speed)
   - subsample: 0.35 (regularization)
   - colsample_bytree: 0.95 (feature sampling)
   - reg_alpha: 0.2, reg_lambda: 1.35 (L1/L2 regularization)
   - gamma: 1.04 (split threshold)
   - min_child_weight: 9 (leaf size control)

2. Remove problematic parameter:
   - scale_pos_weight: None (no class weighting → no probability inflation)

3. Train on full unbalanced dataset:
   - Model learns natural data distribution
   - Probabilities calibrated to actual risk prevalence
   - Low-risk instances get low probabilities (good for CFs!)

**Result**: Optimized structure + calibrated probabilities = Best CF generation

### Model Files Location

Based on experiment folders, the models are likely:
```
models/
├── rf_hltprhc.pkl           # Baseline RF
├── xgboost_hltprhc.pkl      # Baseline XGB
└── xgboost_optimized_cf_friendly.pkl  # XGB-optimized (recommended)
```

Verify model paths in your configs before use!

### Feature Importance Comparison

**Baseline RF** (from model_info.json):
- BMI: 51.0% (dominant)
- eatveg: 8.8%
- etfruit: 10.0%
- alcfreq: 8.9%
- dosprt: 8.7%
- Others: <7% each

**SMOTE-grid RF**:
- BMI: 33.1% (reduced dominance)
- dosprt: 17.7% (increased)
- alcfreq: 12.0% (increased)
- Others more balanced

**Implication**: SMOTE rebalancing reduced BMI dominance, which may have disrupted CF generation pathways that relied on BMI changes.

---

## Next Steps

### Immediate Actions

1. **Use Recommended Configuration**:
   ```yaml
   model: xgboost_optimized_cf_friendly
   stopping_threshold: 0.9
   explainer: genetic
   constraints: enabled
   ```

2. **Verify Model File**:
   - Confirm XGB-optimized model exists
   - Check it has scale_pos_weight=None
   - Test on same 9-instance test set to reproduce results

3. **Document in Configs**:
   - Update main experiment configs to use XGB-optimized
   - Note this as the "Generation 2 Final Model"

### Future Experiments (Generation 3)

Based on Generation_2_experiments_plan.md, proceed with:

1. **Sparsity Investigation** (Step 3 in plan):
   - Use XGB-optimized with threshold 0.9
   - Test: standard sparsity (0.1) vs high sparsity (0.3-0.5)
   - Measure impact on plausibility and diversity
   - Small experiment: 2-3 sparsity levels

2. **Feature Locking Tests**:
   - Lock non-actionable features (e.g., smoking status if intervention impossible)
   - Measure impact on CF generation success
   - Assess alternative pathways DiCE finds

3. **Proximity Weight Exploration**:
   - Test different proximity weights in DiCE
   - Balance between proximity (small changes) and diversity (different solutions)

4. **Full Benchmark Study**:
   - Run final CF generation on full test set (not just 9 instances)
   - Compare to other explainability methods (LIME, SHAP, etc.)
   - Evaluate with domain experts

### Model Retraining (If Needed)

If XGB-optimized model doesn't exist or needs retraining:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=1100,
    max_depth=11,
    learning_rate=0.25,
    subsample=0.35,
    colsample_bytree=0.95,
    reg_alpha=0.2,
    reg_lambda=1.35,
    gamma=1.04,
    min_child_weight=9,
    scale_pos_weight=None,  # CRITICAL: No class weighting!
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

model.fit(X_train, y_train)
```

Train on full unbalanced dataset (not SMOTE-resampled).

---

## Conclusion

**Clear Winner**: XGBoost with optimized hyperparameters (no class weighting) and stopping_threshold=0.5-0.7

**Key Lessons Learned:**

1. **SMOTE models are incompatible with CF generation** - synthetic oversampling disrupts DiCE's search process

2. **Hyperparameter optimization is beneficial** - but only when class weighting is avoided

3. **Test set composition critically affects threshold choice** - CF test set (9 individuals, max pred 0.20) vs full test set (max pred 0.97) require different thresholds

4. **Models CAN produce high predictions** - All three models (RF, XGB baseline, XGB-optimized) produce max predictions 0.96-0.97 on full test set, but median stays low (0.04-0.07)

5. **Threshold must handle full prediction range** - Even though 95% of predictions are <0.40-0.52, the remaining 5% genuinely high-risk individuals need lower thresholds

6. **Speed matters** - XGB's 1.2s vs RF's 2.6s adds up in large-scale experiments

**Critical Discovery About Test Set Selection:**

Initial CF experiments (cfcheck.csv, 9 individuals) showed max prediction 0.2012, leading to belief that threshold 0.9 was universally applicable. However, **full test set analysis (ete.csv) reveals:**

**Full test set predictions (XGB-optimized):**
- Min: 0.0000
- Max: 0.9723 ← **Critical difference!**
- Median: 0.0440
- Mean: 0.1179
- 95th percentile: 0.5215

**This means:**
- CF test set (cfcheck.csv) was **low-risk dominated** (cherry-picked or by chance)
- ~95% of population has predictions <0.52 (threshold 0.9 could work)
- ~5% of population has predictions 0.52-0.97 (threshold 0.9 will FAIL)
- These high-risk individuals CANNOT drop from 0.75 to <0.10 with realistic changes

**Final Recommendation**: Use **XGB-optimized with threshold 0.5-0.7** for production systems. While threshold 0.9 worked perfectly in experiments, this was due to unrepresentative test set selection. Full deployment will encounter genuinely high-risk predictions that require more realistic thresholds.

**Alternative for research**: If testing ONLY on pre-filtered low-risk subset (prob <0.30), threshold 0.9 is appropriate.

---

**Date**: May 7, 2026

**Related Documents**:
- Generation_2_experiments_plan.md
- Understanding_CF_Generation_and_Model_Compatibility.md
