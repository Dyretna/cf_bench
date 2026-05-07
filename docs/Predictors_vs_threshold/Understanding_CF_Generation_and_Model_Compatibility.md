# Understanding Counterfactual Generation and Model Compatibility

**Date**: May 4, 2026
**Purpose**: Comprehensive guide to why model optimization broke CF generation and how to fix it
**Audience**: Researchers working with DiCE without extensive CF/ML background

---

## Table of Contents
1. [What is DiCE and Stopping Threshold?](#what-is-dice-and-stopping-threshold)
2. [The Problem We Discovered](#the-problem-we-discovered)
3. [Why This Happened](#why-this-happened)
4. [The Counterintuitive Truth](#the-counterintuitive-truth)
5. [The Undersampling Alternative](#the-undersampling-alternative)
6. [Solutions and Next Steps](#solutions-and-next-steps)

---

## What is DiCE and Stopping Threshold?

### DiCE (Diverse Counterfactual Explanations)

DiCE is a tool that answers the question: **"What minimal changes to a person's features would improve their health risk prediction?"**

**Example**:
- Person A has 48% risk of cardiovascular disease
- DiCE finds: "If you reduce BMI by 3 points and increase exercise to 3x/week, your risk drops to 8%"
- This is a **counterfactual** - an alternative version where the outcome is better

### Stopping Threshold - The Misunderstood Parameter

**What we thought**: Some vague quality metric about when to stop searching

**What it actually is**: The **target probability** DiCE tries to achieve!

```yaml
stopping_threshold: 0.9
```

This means:
- Target: Get the probability of high-risk (class 1) **below 0.1** (10%)
- In other words: Achieve **90% confidence** that the person is low-risk
- DiCE stops searching when it finds a counterfactual below this threshold

**The formula**:
```
target_probability = 1 - stopping_threshold
```

**Examples**:
- `stopping_threshold: 0.9` → Target prob < 0.1 (10% risk or less) - **Very ambitious**
- `stopping_threshold: 0.7` → Target prob < 0.3 (30% risk or less) - **Moderate**
- `stopping_threshold: 0.5` → Target prob < 0.5 (50% risk or less) - **Just flip prediction**

### Why This Matters

If your model predicts prob = 0.479 (47.9% risk), and `stopping_threshold: 0.9`:
- DiCE needs to find changes that reduce risk from 47.9% → **below 10%**
- That's a reduction of **37.9 percentage points**!
- This might require unrealistic feature changes (lose 15 kg, quit smoking, perfect diet, etc.)

---

## The Problem We Discovered

### Timeline of Events

1. **Baseline models** (April 2026):
   - Random Forest: `class_weight=None`, `max_depth=None`
   - XGBoost: `scale_pos_weight=None`, `max_depth=None`
   - CF generation: ✓ **Worked perfectly** (100% success rate)

2. **Hyperparameter optimization** (April-May 2026):
   - Goal: Improve classification metrics (recall, accuracy)
   - Method: RandomizedSearchCV with 200-700 iterations
   - Result: Found optimal parameters with `class_weight='balanced_subsample'` and `scale_pos_weight=10`
   - Classification: ✓ **Improved** (macro-recall 0.58 → 0.635)

3. **CF generation with optimized models** (May 2026):
   - Same test set (9 instances)
   - Same config (`stopping_threshold: 0.9`)
   - Result: ✗ **Complete failure** (0% success rate)
   - Error: `No counterfactuals found for any of the query points!`

### What Went Wrong?

The optimization improved classification but destroyed the model's probability distribution:

| Metric | Old RF | New RF | Old XGB | New XGB |
|--------|--------|--------|---------|---------|
| **Macro-recall** | 0.58 | **0.635** ✓ | 0.62 | **0.634** ✓ |
| **Avg probability (test set)** | 0.110 | **0.438** | 0.108 | **0.479** |
| **Prob range** | 0.013-0.327 | **0.254-0.606** | 0.026-0.250 | **0.141-0.758** |
| **CF success rate** | 100% ✓ | **0%** ✗ | 100% ✓ | **0%** ✗ |

**Key observation**: Probabilities inflated by **4-6×** on average!

---

## Why This Happened

### Class Weighting: The Double-Edged Sword

Our dataset has **class imbalance**:
- Low-risk (class 0): ~85% of samples
- High-risk (class 1): ~15% of samples

Without correction, models naturally favor the majority class (predict everyone as low-risk).

#### What Class Weighting Does

**Mechanism**:
```python
# Random Forest
class_weight='balanced_subsample'
# Tells the model: "Misclassifying a high-risk person is 5-6× worse than misclassifying a low-risk person"

# XGBoost
scale_pos_weight=10
# Tells the model: "Minority class errors are 10× more important"
```

**Effect on model**:
1. **Shifts decision boundary** toward minority class
2. **Increases predicted probabilities** for most instances
3. **Improves recall** (catches more high-risk people)
4. **Degrades probability calibration** (probabilities no longer reflect actual risk)

#### Before and After Class Weighting

**Test Instance 0** (actual class: low-risk):
- Real risk: Low (~6%)
- Old model prediction: 0.063 (6.3% risk) ← **Calibrated!**
- New model prediction: 0.479 (47.9% risk) ← **Inflated!**

**Test Instance 4** (actual class: low-risk):
- Real risk: Low (~3%)
- Old model prediction: 0.027 (2.7% risk) ← **Calibrated!**
- New model prediction: 0.254 (25.4% risk) ← **Inflated!**

The new model thinks low-risk people have 25-48% risk when they actually have 3-6% risk!

### Why Inflated Probabilities Break CF Generation

Remember `stopping_threshold: 0.9` means "get probability below 0.1".

#### Old Model (Calibrated Probabilities)
```
Person at 0.063 probability:
- Target: below 0.1
- Status: ✓ Already there!
- DiCE: Finds even better CFs (down to 0.02-0.05)
- Feature changes needed: Small (BMI -2, sport +1)
```

#### New Model (Inflated Probabilities)
```
Person at 0.479 probability:
- Target: below 0.1
- Status: ✗ Need to drop by 0.379 (79% reduction)!
- DiCE: Cannot find any valid CF within realistic constraints
- Feature changes needed: Massive (BMI -15, perfect diet, quit smoking, max exercise)
```

**The problem**: You have a limited "feature budget" (realistic changes). That budget is enough for a 0.03 drop, not a 0.38 drop.

---

## The Counterintuitive Truth

### Your Original Intuition (Seemed Logical!)

> "If probability is 0.479 and needs to go to 0.2, that's easier than going from 0.063 to 0.01, right? Both are ~50% relative reductions, but the first involves larger absolute numbers."

**This intuition is correct** for relative changes and feature sensitivity!

### Why It's Actually Wrong for CF Generation

The key insight: **You're not trying to halve the probability. You're trying to reach a fixed target (0.1).**

#### The Speed Limit Analogy

Think of it like driving with a speed limit:

**Scenario A**: Current speed 5 km/h, limit 10 km/h
- Already legal! ✓
- Could go faster if you want

**Scenario B**: Current speed 100 km/h, limit 10 km/h
- Need to slow down by 90 km/h! ✗
- Requires massive braking

**Relative change** (both 50% reduction):
- 100 → 50 km/h: Easy (light braking)
- 5 → 2.5 km/h: Easy (light braking)

**Absolute change** (reach 10 km/h limit):
- From 5 km/h: Need to slow by 0 (or speed up!) ✓
- From 100 km/h: Need to slow by 90! ✗

#### Applied to Our Models

| Starting Prob | Target (< 0.1) | Absolute Change Needed | Feasibility |
|---------------|----------------|------------------------|-------------|
| 0.063 (old)   | 0.1            | 0.0 (already there!)   | ✓ **Easy** |
| 0.147 (old)   | 0.1            | 0.047                  | ✓ **Achievable** |
| 0.254 (new)   | 0.1            | 0.154                  | ? **Hard** |
| 0.479 (new)   | 0.1            | 0.379                  | ✗ **Impossible** |

### The Feature Budget Concept

You can only change features within realistic bounds:
- BMI: Can change by ~5-10 points realistically
- Smoking: Can improve by 1-3 levels
- Diet: Can improve by 1-2 levels
- Exercise: Can increase by 0-3 levels

**Each feature change "buys" a certain probability reduction.**

#### Example: BMI Changes

Let's trace what happens when you reduce BMI from 25 to 15:

**Old model** (starting at prob=0.063):
```
BMI 25 → prob 0.063
BMI 24 → prob 0.058
BMI 23 → prob 0.053
BMI 22 → prob 0.048
BMI 21 → prob 0.043  ← Already below 0.1!
```
Success with only 4 BMI points spent!

**New model** (starting at prob=0.479):
```
BMI 25 → prob 0.479
BMI 24 → prob 0.461
BMI 23 → prob 0.443
BMI 22 → prob 0.425
BMI 21 → prob 0.408
BMI 20 → prob 0.391
BMI 19 → prob 0.375
BMI 18 → prob 0.359
BMI 17 → prob 0.344
BMI 16 → prob 0.329
BMI 15 → prob 0.315  ← Still at 31.5% after maxing out BMI!
```
Failure even after spending all 10 BMI points!

**And** you still have 7 other features to optimize to their extremes, which might still not be enough!

### Why Lower Probabilities Are Actually Better

**Low starting probability** (0.01-0.15):
- ✓ Already close to or below target
- ✓ Small feature changes sufficient
- ✓ Realistic recommendations possible
- ✓ DiCE finds multiple diverse CFs

**High starting probability** (0.3-0.7):
- ✗ Far from target
- ✗ Massive feature changes required
- ✗ Would need unrealistic extreme values
- ✗ DiCE cannot find any valid CF

**Counterintuitive conclusion**: Being "confident" (low prob for low-risk people) is better than being "uncertain" (mid-range probs for everyone)!

---

## The Undersampling Alternative

### What is Undersampling?

Instead of using class weights, you can balance the dataset by:
- **Keeping all minority class samples** (high-risk: ~15%)
- **Randomly removing majority class samples** (low-risk: remove ~70% to balance)
- Training on the balanced dataset

### Why It Was Considered

From `notebooks/predictors/optimizing_predictors/undersampling_strategy/README.md`:

The undersampling experiments tested whether balanced training data would:
1. Improve recall (like class weighting)
2. Maintain probability calibration (unlike class weighting)
3. Preserve CF generation capability

### The Results

**Model performance** (`01_rf_performance_balanced_training_data.ipynb`):
- Balanced training improved recall slightly
- But: threw away 70% of training data
- Result: Worse overall performance than class weighting

**CF generation** (`02_cfs_on_balanced_vs_unbalanced.ipynb`):
- Balanced models: Better CF success than weighted models
- But: Still worse than unweighted models on full dataset
- Reason: Smaller training set → less robust model

### Why Undersampling Didn't Work

**Trade-offs**:
1. ✓ Avoids probability inflation (better than class weighting)
2. ✗ Throws away valuable training data
3. ✗ Model has less information to learn from
4. ✗ Overfits to minority class
5. ✗ Still shifts decision boundary (just less aggressively)

**Conclusion**: Undersampling is better than class weighting for CF generation, but worse than training on the full unbalanced dataset without any correction.

### Why Full Dataset Wins for CF Generation

**Unbalanced dataset** (no correction):
- Model learns natural data distribution
- Probabilities calibrated to actual prevalence
- Low-risk people get low probabilities (good for CFs!)
- Trade-off: Lower recall (misses some high-risk cases)

**For CF generation**: We want the model to be **confident** in its predictions, not **balanced** in its errors.

---

## Solutions and Next Steps

### Immediate Solution: Use Baseline Models

For continued CF experiments, use the old models:

```yaml
# configs/your_experiment.yaml
model_path: "models/rf_hltprhc.pkl"       # Random Forest
# OR
model_path: "models/xgboost_hltprhc.pkl"  # XGBoost

stopping_threshold: 0.9  # Can use ambitious targets
```

These models:
- ✓ Generate CFs successfully
- ✓ Have calibrated probabilities
- ✗ Lower recall (~58-62% macro-recall)

### Long-Term Solution: Train New Models

Train models optimized for CF generation, not just classification:

#### Model Training Strategy

**Keep the good hyperparameters**, remove the problematic ones:

```python
# Random Forest - CF-Friendly Configuration
RandomForestClassifier(
    n_estimators=450,           # ✓ Keep (from optimization)
    max_depth=6,                # ✓ Keep (from optimization)
    min_samples_leaf=3,         # ✓ Keep (from optimization)
    min_samples_split=2,        # ✓ Keep (from optimization)
    max_features='sqrt',        # ✓ Keep (from optimization)
    class_weight=None,          # ✗ Remove! (causes probability inflation)
    random_state=42
)
```

```python
# XGBoost - CF-Friendly Configuration
XGBClassifier(
    n_estimators=450,           # ✓ Keep
    max_depth=4,                # ✓ Keep
    learning_rate=0.05,         # ✓ Keep
    subsample=0.75,             # ✓ Keep
    colsample_bytree=0.70,      # ✓ Keep
    min_child_weight=3,         # ✓ Keep
    gamma=1.2,                  # ✓ Keep
    reg_lambda=2.0,             # ✓ Keep
    reg_alpha=1.0,              # ✓ Keep
    scale_pos_weight=None,      # ✗ Remove! (causes probability inflation)
    random_state=42
)
```

**Key principle**: Take the optimized structure (tree depth, regularization) but train without class imbalance correction.

#### Expected Results

| Metric | Old Baseline | Optimized (weighted) | New (CF-friendly) |
|--------|-------------|---------------------|-------------------|
| Macro-recall | 0.58 | 0.635 | ~0.60-0.62 |
| Probability range | 0.01-0.35 | 0.25-0.75 | ~0.01-0.40 |
| CF success rate | 100% | 0% | ~95-100% |

You'll get:
- ✓ Better than baseline classification (improved structure)
- ✓ CF generation capability (calibrated probabilities)
- ≈ Slightly lower recall than weighted models (acceptable trade-off)

### Alternative Approaches (For Future Work)

#### 1. Probability Calibration (Post-Training)

Train with class weights, then fix the probabilities:

```python
from sklearn.calibration import CalibratedClassifierCV

# Train optimized model (with class weights)
base_model = RandomForestClassifier(class_weight='balanced_subsample', ...)
base_model.fit(X_train, y_train)

# Calibrate probabilities on validation set
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='isotonic',  # or 'sigmoid'
    cv='prefit'
)
calibrated_model.fit(X_val, y_val)

# Now use calibrated_model for CF generation
```

**Pros**: Keeps high recall + restores calibration
**Cons**: Requires validation set, adds complexity, not guaranteed to fully restore CF capability

#### 2. Threshold Optimization (Instead of Class Weighting)

```python
# Train unweighted model
model = RandomForestClassifier(class_weight=None, ...)
model.fit(X_train, y_train)

# Find optimal classification threshold
from sklearn.metrics import precision_recall_curve
probs = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, probs)
# Choose threshold that maximizes F1 or achieves target recall
optimal_threshold = 0.35  # Example (instead of default 0.5)

# Classify with custom threshold
predictions = (probs > optimal_threshold).astype(int)
```

**Pros**: Improves recall without touching probabilities
**Cons**: Less improvement than class weighting, needs threshold tuning per use case

#### 3. Separate Models (Best of Both Worlds)

```python
# Model 1: Classification (with class weights)
classification_model = load('models/rf_optimized_hltprhc.pkl')
predictions = classification_model.predict(X)  # Use for actual predictions

# Model 2: CF Generation (without class weights)
cf_model = load('models/rf_cf_friendly_hltprhc.pkl')
counterfactuals = dice.generate_counterfactuals(
    model=cf_model,  # Use this for explanations
    query_instances=X
)
```

**Pros**: Each model optimized for its specific task
**Cons**: Need to train/maintain two models, explanations based on different model

### Recommended Next Steps

1. **Train new CF-friendly models** (remove class weighting, keep other optimal params)
2. **Test CF generation** on the same 9-instance test set
3. **Validate probabilities** are calibrated (check distribution is similar to baseline)
4. **Compare metrics**:
   - Classification: Accuracy, recall, precision, F1
   - CF generation: Success rate, average feature changes, validity
5. **If successful**: Use these models for your benchmarking experiments

---

## Key Takeaways

### For Understanding DiCE

1. **Stopping threshold** = target probability, not a vague quality metric
   - `0.9` means "get probability below 0.1 (10% risk)"
   - `0.5` means "just flip the prediction"

2. **Lower probabilities make CF generation easier**, not harder
   - Starting at 0.06 → need to reach 0.1: Already there!
   - Starting at 0.48 → need to reach 0.1: Impossible!

3. **DiCE expects calibrated probabilities**
   - Model output should reflect actual risk
   - Class weighting breaks this assumption

### For Model Training

1. **Class weighting improves classification but breaks CF generation**
   - Inflates probabilities by 4-10×
   - Makes target thresholds unreachable

2. **Undersampling also problematic**
   - Better than class weighting for CFs
   - But worse than full dataset without correction

3. **Best approach for CF generation**:
   - Train on full unbalanced dataset
   - No class weighting or undersampling
   - Keep optimized structure (tree depth, regularization)
   - Accept slightly lower recall

### For Your Research

- Use baseline models for current experiments
- Train new CF-friendly models (optimized structure, no class weights)
- Expected: ~60% macro-recall, 95-100% CF success
- This balances classification performance with CF capability

---

## Glossary

**Calibrated Probabilities**: Model output reflects actual risk (e.g., "60% probability" means 60% of similar cases are high-risk)

**Class Imbalance**: One class has many more samples than another (85% low-risk, 15% high-risk)

**Class Weighting**: Technique to handle imbalance by penalizing errors on minority class more heavily

**Counterfactual**: Alternative version of a person's features that would lead to a better outcome

**Feature Budget**: The realistic range of changes possible for each feature (e.g., can only lose 10kg BMI)

**Macro-recall**: Average recall across both classes (good for imbalanced datasets)

**Probability Inflation**: When class weighting artificially increases predicted probabilities beyond actual risk

**Stopping Threshold**: DiCE parameter specifying target probability to achieve (target = 1 - threshold)

---

## References
- **Optimization experiments**: See `notebooks/predictors/optimizing_predictors/`
- **Undersampling experiments**: See `notebooks/predictors/optimizing_predictors/undersampling_strategy/`
---

**Document Version**: 1.0
**Last Updated**: May 4, 2026
