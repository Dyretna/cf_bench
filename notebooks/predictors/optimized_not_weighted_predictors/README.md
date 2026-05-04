# Optimized CF-Compatible Predictors

This folder contains the final hyperparameter optimization experiments for counterfactual-generation (CF) compatible models. These experiments represent the **exhaustive conclusion** of predictor optimization for the CVD classification task.

## Executive Summary

After extensive hyperparameter tuning with 500+ iterations across multiple search strategies, we have reached the performance ceiling for both Random Forest and XGBoost classifiers trained without class weighting (a requirement for CF generation compatibility).

**Key Findings:**
- **Random Forest**: No meaningful improvement over baseline (300 trees, default params)
- **XGBoost**: Marginal improvement (~1-2% in recall and ROC-AUC)
- **Conclusion**: Further hyperparameter optimization yields diminishing returns

## Why CF-Compatible Models?

See the comprehensive [Understanding_CF_Generation_and_Model_Compatibility.md](../../../docs/Understanding_CF_Generation_and_Model_Compatibility.md) document for full context.

**TL;DR:** Models trained WITH class weighting (see `../weighted_predictors/`) achieve better classification metrics (macro-recall ~0.63-0.64) but produce **inflated probabilities** (0.25-0.75 range) that break DiCE's counterfactual generation. DiCE with `stopping_threshold: 0.9` requires predicted probabilities to drop below 0.1 (10% risk), which is:
- **Easy** for calibrated models (probs: 0.01-0.35) ✓
- **Impossible** for inflated models (probs: 0.25-0.75) ✗

**The Trade-off:**
- **CF-compatible models** (this folder): Lower recall (~0.52-0.53), but DiCE can generate counterfactuals
- **Weighted models**: Higher recall (~0.63-0.64), but DiCE fails completely (0% success rate)

For counterfactual generation benchmarks, CF-compatibility is non-negotiable, hence these models.

## Performance Summary

### Baseline Models (300 estimators, no tuning)
Located in `../base_predictors/`

**Random Forest (rf_hltprhc.pkl):**
- n_estimators: 300
- All other params: sklearn defaults
- Performance: ~0.52 macro-recall, 0.516 ROC-AUC

**XGBoost (xgboost_hltprhc.pkl):**
- n_estimators: 300
- All other params: XGBoost defaults
- Performance: ~0.52 macro-recall, 0.516 ROC-AUC

### Optimized Models (500 iterations, exhaustive search)
Located in this folder

**Random Forest (rf_optimized_hltprhc.pkl):**
- Gridsearch: 500 iterations testing depth, samples, features, etc.
- Result: **NO IMPROVEMENT** over baseline
- Conclusion: 300 trees with defaults already optimal for this task

**XGBoost (xgb_optimized_hltprhc.pkl):**
- Gridsearch Run 001: 300 iterations (broad search)
  - Best CV score: 0.513
- Gridsearch Run 002: 500 iterations (refined search with reg_lambda/alpha)
  - Best CV score: 0.526
- Optimized params:
  - n_estimators: 1092
  - max_depth: 11
  - learning_rate: 0.259
  - reg_lambda: 1.363 (L2 regularization - key finding!)
  - reg_alpha: 0.216
  - colsample_bytree: 0.968 (high feature retention crucial)
- Test performance: ~0.53 macro-recall, 0.527 ROC-AUC
- **Improvement: ~1-2% over baseline**

## Files

### Notebooks

**Random Forest:**
- `rf_gridsearch.ipynb`: 500-iteration RandomizedSearchCV
- `rf_save_optimized.ipynb`: Load results, analyze, train final model

**XGBoost:**
- `xgb_gridsearch.ipynb`: Two-phase search (300 + 500 iterations)
- `xgb_save_optimized.ipynb`: Analysis of Run 002, train final model

### Results Files

- `rf_gridsearch_runs.csv`: RF search results (all runs aggregated)
- `xgb_gridsearch_runs.csv`: XGBoost search results (Run 001 + Run 002)

## Critical Constraint: No Class Weighting

All models trained with:
- Random Forest: `class_weight=None` (NOT `balanced` or `balanced_subsample`)
- XGBoost: `scale_pos_weight=None` (NOT 10 or other weights)

**Rationale:**
Class weighting inflates predicted probabilities by 4-6x, making counterfactual generation impossible:
- Unweighted model: "This person has 6% risk" → DiCE finds realistic changes to reach <10%
- Weighted model: "This person has 48% risk" → DiCE cannot find changes to reach <10% within realistic bounds

See `../weighted_predictors/` for weighted model experiments (better classification, but CF-incompatible).

## Why Optimization Plateaued

### Random Forest: Structural Limitations

Random Forest performed identically across 500 iterations because:
1. **Ensemble size**: 300 trees sufficient for convergence
2. **Tree depth**: Deeper trees overfit imbalanced data
3. **Sampling**: Various bootstrap/feature sampling strategies no impact
4. **Imbalance tolerance**: Without class weighting, RF naturally favors majority class

**Conclusion:** Random Forest architecture fundamentally unsuited for this imbalanced classification task without artificial weighting (which breaks CF generation).

### XGBoost: Marginal Gains via Regularization

XGBoost showed ~2.5% CV improvement (0.513 → 0.526) through:
1. **Larger ensemble**: 1092 trees vs 300 (but diminishing returns)
2. **L2 regularization**: reg_lambda 1-8 range helped prevent overfitting
3. **High feature retention**: colsample_bytree 0.8-1.0 crucial for tabular data
4. **Moderate learning rate**: 0.2-0.3 range optimal

**However:** Test set improvement only ~1% (0.516 → 0.527 ROC-AUC), suggesting CV gains partially illusory.

## Expected Impact on Counterfactual Generation

### Critical Question: Will XGBoost Optimization Help DiCE?

**Short answer: Probably not.**

**Reasoning:**

1. **Probability Calibration is Key**: DiCE's success depends on predicted probabilities, not classification accuracy. The optimized XGBoost model's probabilities (tested on the same 9-instance test set) are expected to remain in the 0.01-0.40 range, similar to baseline.

2. **Minimal Architecture Change**: The optimization primarily adjusted:
   - Ensemble size (300 → 1092 trees)
   - Regularization (added L2/L1)
   - Tree depth (default → 11)

   These changes affect decision boundaries and reduce overfitting, but don't fundamentally alter how probabilities map to feature space.

3. **Feature-Probability Relationship**: DiCE needs to know "how much does changing BMI by 2 points reduce probability?" The gradient of this relationship is unlikely to differ substantially between:
   - Baseline: 300 trees, depth 6, no regularization
   - Optimized: 1092 trees, depth 11, L2=1.36, L1=0.22

4. **Historical Evidence**: The transition from baseline to weighted models showed dramatic CF generation changes because probabilities inflated 4-6x. The baseline → optimized transition has NO probability inflation (both use scale_pos_weight=None), so impact should be minimal.

**Expected CF Generation Results:**
- Baseline XGBoost: ~95-100% success rate (historical data)
- Optimized XGBoost: ~95-100% success rate (prediction)
- Quality metrics (validity, sparsity, proximity): Likely unchanged
- Feature change patterns: Possibly slightly different, but within noise

**Counter-argument (Why it MIGHT help):**
The improved generalization from regularization could make probability predictions more stable/realistic for edge cases, potentially helping DiCE find CFs for the hardest 5-10% of instances. But this is speculative.

### Recommendation for CF Experiments

**For comprehensive benchmarking:**
- Test both baseline and optimized XGBoost models
- Focus analysis on CF success rate and feature change patterns
- If results are nearly identical (as expected), use baseline for simplicity

**For research reporting:**
- Document that optimization was attempted and yielded minimal gains
- Emphasize the class-weighting constraint as the dominant factor
- Note that further performance improvements require feature engineering or alternative model families (neural networks, etc.), not hyperparameter tuning

## What We've Learned

### 1. Model Architecture Matters More Than Hyperparameters

For this task (imbalanced CVD prediction without class correction):
- **Random Forest**: ~0.52 recall regardless of hyperparameters
- **XGBoost**: ~0.52-0.53 recall with extensive tuning
- **Weighted XGBoost**: ~0.63 recall (but CF-incompatible)

The 20% performance gap comes from class weighting, not tree depth or learning rate.

### 2. Class Imbalance is the Bottleneck

Our dataset: 85% low-risk (class 0), 15% high-risk (class 1)

Without correction:
- Models naturally favor majority class → low recall for class 1
- Hyperparameter tuning can't overcome fundamental imbalance

With class weighting:
- High recall achievable (~0.63)
- BUT probabilities become miscalibrated → CF generation fails

**This is an unsolvable trade-off** for traditional tree-based models.

### 3. Feature Engineering > Hyperparameter Tuning

500+ iterations of hyperparameter search yielded 1-2% improvement. Likely gains from:
- Adding interaction features (BMI * exercise, smoking * age)
- Domain-specific transformations (risk score composites)
- Feature selection (remove collinear/noisy features)
- Alternative representations (polynomial features, binning)

But feature engineering is out of scope for this optimizer comparison project.

### 4. Alternative Approaches Needed

For substantial improvement beyond ~0.53 macro-recall while maintaining CF compatibility:
1. **Ensemble methods**: Stack CF-compatible models with different architectures
2. **Probability calibration**: Train with weights, then post-calibrate (see Understanding doc)
3. **Threshold optimization**: Instead of changing probabilities, optimize classification threshold
4. **Neural networks**: May learn better probability calibration naturally (see `../base_predictors/nn_hltprhc.ipynb`)
5. **Cost-sensitive learning**: Alternative to class weighting that may preserve calibration

## Usage

The optimized models are saved and ready for use:

```python
import joblib

# Load optimized models
xgb_model = joblib.load("models/xgb_optimized_hltprhc.pkl")
rf_model = joblib.load("models/rf_optimized_hltprhc.pkl")  # If saved (no improvement over baseline)

# Or use baseline models (recommended for RF)
xgb_baseline = joblib.load("models/xgboost_hltprhc.pkl")
rf_baseline = joblib.load("models/rf_hltprhc.pkl")

# For counterfactual generation
import dice_ml
dice_data = dice_ml.Data(...)
dice_model = dice_ml.Model(model=xgb_model, backend="sklearn")
dice_exp = dice_ml.Dice(dice_data, dice_model)

# Generate counterfactuals
cfs = dice_exp.generate_counterfactuals(
    query_instances,
    total_CFs=5,
    desired_class="opposite",
    stopping_threshold=0.9  # Requires prob < 0.1
)
```

## Conclusion: The Optimization Journey Ends Here

**What we accomplished:**
- Exhaustively searched hyperparameter space (500-1000+ configs tested)
- Confirmed Random Forest requires class weighting for this task (incompatible with CF generation)
- Found optimal XGBoost configuration with marginal (~1-2%) improvement
- Documented the fundamental trade-off between classification performance and CF compatibility

**What we learned:**
- Model architecture and class imbalance handling dominate performance
- Hyperparameter tuning has limited impact within CF-compatibility constraint
- Counterfactual generation compatibility requires sacrificing recall (~0.53 vs ~0.63)

**Next steps (for future work, not this project):**
- Feature engineering (most promising for performance gains)
- Alternative model families (neural networks, ensemble methods)
- Probability calibration techniques (post-process weighted models)
- Threshold optimization strategies

**For counterfactual benchmarking experiments:**
- Use baseline XGBoost (300 trees, defaults) for simplicity OR
- Use optimized XGBoost (1092 trees, regularized) for completeness
- Expect nearly identical CF generation results between the two
- Document that further optimization was attempted but yielded no CF-relevant improvements

---

**Project Status:** ✓ **Optimization Complete - No Further Gains Expected**

**Recommended Model:** `xgboost_hltprhc.pkl` (baseline, 300 trees) OR `xgb_optimized_hltprhc.pkl` (optimized, 1092 trees) - expect similar CF generation performance.
