# Stopping Threshold in DiCE - Overview

This document explains the **stopping_threshold** parameter in DiCE's counterfactual generation process. It covers what it is, why it matters, and how to choose appropriate values for your use case.

---

## 1. What Is Stopping Threshold?

The `stopping_threshold` parameter in DiCE defines **when to stop searching for counterfactuals** based on the predicted probability of the desired outcome class.

**Key Concept:**
```
stopping_threshold = 1 - target_probability
```

**In plain terms:**
- DiCE keeps searching until it finds counterfactuals where the model's predicted probability for the undesired class falls **below** `1 - stopping_threshold`
- Higher stopping_threshold = more ambitious target (lower target probability)
- Lower stopping_threshold = easier target (higher target probability allowed)

---

## 2. Understanding the Math

### The Formula

If you set `stopping_threshold = 0.9`, DiCE will try to find counterfactuals where:

```
P(high-risk class) < 1 - 0.9 = 0.1
```

This means: "Find a version of this person where their risk is below 10%"

### Common Values

| stopping_threshold | Target Probability | Meaning |
|-------------------|-------------------|---------|
| 0.9 | < 0.1 (10%) | Very ambitious - reduce risk to under 10% |
| 0.7 | < 0.3 (30%) | Moderate - reduce risk to under 30% |
| 0.5 | < 0.5 (50%) | Basic - just flip the prediction |
| 0.3 | < 0.7 (70%) | Very easy - allow up to 70% risk |
| 0.1 | < 0.9 (90%) | Minimal - barely any improvement needed |

---

## 3. Why Stopping Threshold Matters

### For Users (Domain Experts)

**Clinical/Practical Perspective:**
- Higher thresholds (0.8-0.9) generate recommendations that lead to **meaningful risk reduction**
- Lower thresholds (0.3-0.5) may produce changes that barely help
- The threshold should match **real-world goals** (e.g., "reduce cardiovascular risk to low category")

**Example - Health Risk:**
- Original: 45% risk of heart disease
- With threshold 0.9: Find changes → <10% risk (major lifestyle overhaul)
- With threshold 0.5: Find changes → <50% risk (minimal improvement, still "medium risk")

### For Machine Learning

**Technical Perspective:**
- Stopping threshold controls the **optimization target** for DiCE's search algorithm
- Higher threshold = **harder optimization problem** (may fail to find valid CFs)
- Lower threshold = **easier optimization problem** (more likely to succeed)
- Success depends on **how far** the original instance is from the target

---

## 4. The Probability Distance Problem

### Why Starting Probability Matters

The key insight: **It's not about relative change, it's about absolute distance to target.**

#### Scenario A: Low Starting Probability (Well-Calibrated Model)

```
Original probability: 0.06 (6% risk)
Threshold 0.9 → Target: <0.1 (10% risk)
Distance to target: 0 (already there!)
Result: ✓ Easy - DiCE finds many valid CFs
```

#### Scenario B: High Starting Probability (Inflated Probabilities)

```
Original probability: 0.48 (48% risk)
Threshold 0.9 → Target: <0.1 (10% risk)
Distance to target: 0.38 (need to reduce by 38 percentage points!)
Result: ✗ Hard/Impossible - DiCE may find no valid CFs
```

### The Feature Budget Concept

You can only change features within realistic bounds. Each feature change "buys" a certain amount of probability reduction.

**Example with threshold 0.9 (target <10%):**

**Starting at 6% probability:**
```
BMI 25 → 23: prob goes 0.06 → 0.04
Sport 0 → 2: prob goes 0.04 → 0.02
✓ Success with just 2 small changes
```

**Starting at 48% probability:**
```
BMI 25 → 20: prob goes 0.48 → 0.35 (not enough!)
Sport 0 → 7: prob goes 0.35 → 0.28 (still not enough!)
Smoking 3 → 6: prob goes 0.28 → 0.22 (still not enough!)
Diet 3 → 7: prob goes 0.22 → 0.18 (still not enough!)
✗ Even extreme changes can't reach <10%
```

---

## 5. Choosing the Right Threshold

The optimal threshold depends on several factors:

### 1. Model Training Strategy and Prediction Distribution (MOST CRITICAL)

**This is the most important factor - you MUST understand your model's actual prediction range before choosing a threshold.**

#### Models Trained on Imbalanced Data (No Class Weighting)

**Characteristics:**
- Trained without class_weight or scale_pos_weight
- Naturally imbalanced dataset (e.g., 10% positive class)
- Prediction distribution: RIGHT-SKEWED but NOT compressed

**Typical prediction distribution (corrected based on full test set):**
- Maximum predictions: **0.95-0.98** (full range possible!)
- Median predictions: 0.04-0.07 (most users low-risk)
- 95th percentile: 0.40-0.52 (outliers exist)
- Mean: 0.10-0.12

**Recommended threshold: 0.5-0.7 (NOT 0.9!)**

**Why threshold 0.9 fails in production:**
- While MOST users have low predictions (<0.30), ~5% are genuinely high-risk
- Person with 0.75 probability CANNOT drop to <0.10 with realistic changes
- Requires 0.65 reduction - impossible even with extreme lifestyle modifications
- These high-risk individuals will get "No counterfactuals found" errors

**Real example from XGBoost-optimized model (CORRECTED):**

**Initial CF test (misleading):**
- Test set: 9 individuals from cfcheck.csv
- Max prediction: 0.2012
- Threshold 0.9 achieved 100% success
- Conclusion: "Threshold 0.9 works perfectly!"

**Full test set analysis (reality):**
- Test set: Full ete.csv dataset
- Max prediction: **0.9723** ← Critical difference!
- Median: 0.0440, Mean: 0.1179
- 95th percentile: 0.5215
- Conclusion: **CF test set was unrepresentative (low-risk dominated)**

**Critical insight:** The CF test set happened to contain only low-risk individuals, making threshold 0.9 appear universally applicable. Full deployment will encounter genuinely high-risk predictions requiring threshold 0.5-0.7.

#### Well-Calibrated Models (Balanced Training or Class Weighting)

**Characteristics:**
- Trained with class_weight='balanced' or scale_pos_weight
- Or trained on balanced/SMOTE data
- Model predictions spread across FULL range

**Typical prediction distribution:**
- Predictions range from 0.0 to 1.0
- High-risk individuals get predictions 0.60-0.90
- Low-risk individuals get predictions 0.05-0.20

**Recommended threshold: 0.5-0.7**

**Why lower threshold needed:**
- Person with 0.75 probability CANNOT drop to <0.10 realistically
- Would require 0.65 reduction - impossible with feasible changes
- Threshold 0.5 allows dropping from 0.75 to <0.375 (more realistic)

**Warning:** These models often have **inflated probabilities** that make CF generation harder!

### 2. How to Check Your Model's Prediction Distribution

**BEFORE choosing a threshold, run this check:**

```python
# Get predictions on representative test data
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Min prediction: {y_pred_proba.min():.3f}")
print(f"Max prediction: {y_pred_proba.max():.3f}")
print(f"Mean prediction: {y_pred_proba.mean():.3f}")
print(f"Median prediction: {np.median(y_pred_proba):.3f}")
print(f"95th percentile: {np.percentile(y_pred_proba, 95):.3f}")
```

**Decision rules (CORRECTED):**
- If max < 0.35: Use threshold 0.7-0.9 (low-risk population only)
- If max 0.35-0.60: Use threshold 0.6-0.7
- If max > 0.60: Use threshold 0.5-0.6 ✓ (most realistic)

**WARNING:** Even if max is 0.97 but 95th percentile is <0.50, you might be tempted to use threshold 0.9. DON'T! That remaining 5% of high-risk users will fail completely. Use threshold based on MAX, not percentile.

### 3. Domain Requirements
- Medical/Clinical: Match threshold to model's prediction distribution (see section 5.2)
- Exploratory analysis: Test multiple thresholds to understand trade-offs
- Proof of concept: Use conservative threshold based on model check

### 4. Practical Constraints
- Limited actionable features: May need lower threshold
- Many actionable features: Can use higher threshold if model supports it

### Recommended Approach

**Step 1: Check your model's prediction distribution (REQUIRED)**

```python
# Analyze model predictions
y_pred = model.predict_proba(X_test)[:, 1]
max_pred = y_pred.max()

print(f"Max prediction: {max_pred:.3f}")

# Decision tree
if max_pred < 0.35:
    recommended_threshold = 0.9
    print("✓ Conservative model - use threshold 0.9")
elif max_pred < 0.60:
    recommended_threshold = 0.7
    print("⚠ Moderate range - use threshold 0.7")
else:
    recommended_threshold = 0.5
    print("⚠ Wide range - use threshold 0.5-0.6")
```

**Step 2: Test with representative queries**

```python
# Test on sample of real users
exp.generate_counterfactuals(
    query_instance=sample_users,
    stopping_threshold=recommended_threshold
)

# Check success rate
if success_rate < 85%:
    # Lower threshold
    recommended_threshold -= 0.2
```

**Common mistake:** Choosing threshold based on "how strict you want to be" without checking if model can actually produce predictions in that range!

### Key Misconceptions Debunked

#### Misconception 1: "Threshold depends on population composition"

**Wrong!** It depends on **model's prediction distribution**, not population composition.

**Example that proves this:**
- Test set: 9 individuals, 2 have CVD (22% positive rate - mixed population)
- Model predictions: max 0.2012, most <0.10
- Threshold 0.9 worked perfectly (100% success)
- **Why?** Model trained on imbalanced data (no class weights) compressed ALL predictions to <0.30, regardless of true labels

**The truth:**
- Population composition affects TRAINING (class imbalance in training data)
- Training strategy affects MODEL BEHAVIOR (prediction distribution)
- Model behavior determines THRESHOLD choice
- A "mixed" test population with sick AND healthy people can still have low prediction range if model is conservative

#### Misconception 2: "Lower threshold = easier to achieve"

**Actually:** The TARGET is easier, but you're asking for less meaningful change.

- Threshold 0.1 (target <90%) is easy to achieve but barely helps
- Threshold 0.9 (target <10%) is harder but provides real risk reduction

#### Misconception 3: "High-risk people can't reach low thresholds"

**It depends on the MODEL, not the person's true risk!**

**Example:**
- Person has CVD (true positive)
- Conservative model predicts: 0.18 probability
- With threshold 0.9: Need to drop from 0.18 → <0.10 (only 0.08 reduction)
- Result: ✓ Successfully generated CFs

**The key:** If model never predicts above 0.30, then threshold 0.9 works for EVERYONE, including people with actual disease. The model's conservatism makes aggressive thresholds feasible.

#### Misconception 4: "Higher starting probability = easier to reduce"

**Wrong!** Higher starting probability makes it **harder** because:
- You need to cross more absolute distance
- You have a limited "feature budget" (realistic changes)
- Distance to target matters, not relative change

**Correct thinking:**
- Low prob (0.06) with threshold 0.9: Need to drop by 0 → EASY
- High prob (0.48) with threshold 0.9: Need to drop by 0.38 → VERY HARD

#### Misconception 5: "Threshold is a quality metric"

**Actually:** It's a **search target**, not a quality measure.

- It doesn't measure how good the CF is
- It defines what "valid" means for stopping the search
- Quality is measured separately (sparsity, proximity, diversity, etc.)

---

## 7. Threshold in Different Scenarios

### Scenario 1: Imbalanced Training, No Class Weighting (Production Deployment) ✓ RECOMMENDED

**Characteristics:**
- Model trained without class weighting on imbalanced data
- Right-skewed predictions: median low (0.04-0.07) but max high (0.95-0.98)
- Most users low-risk, but ~5% genuinely high-risk

**Recommended threshold: 0.5 (primary) or 0.7 (ambitious)**

**Real-world example - CORRECTED (XGBoost-optimized):**

```python
# Initial CF test (MISLEADING):
# - 9 individuals from cfcheck.csv
# - Max prediction: 0.2012
# - Threshold 0.9 worked perfectly

# Full test set analysis (REALITY):
# - Full ete.csv dataset
# - Max prediction: 0.9723 ← NOT 0.20!
# - Median: 0.0440, 95th percentile: 0.5215
# - Conclusion: CF test was unrepresentative

# CORRECT approach for production:
exp.generate_counterfactuals(
    query_instance=patient,
    stopping_threshold=0.5  # Target <50% of original
)

# Why 0.5 instead of 0.9:
# - Low-risk patient (0.10 prob) → target <0.05 ✓ achievable
# - High-risk patient (0.75 prob) → target <0.375 ✓ achievable
# - With threshold 0.9: 0.75 → <0.075 ✗ IMPOSSIBLE
```

**Why threshold 0.9 FAILS in production:**
- CF test set (9 individuals) happened to be low-risk only
- Full dataset contains ~5% high-risk predictions (0.52-0.97)
- Person with 0.75 prob needs 0.675 drop to reach <0.075
- This is impossible with realistic feature changes
- These users will get "No counterfactuals found" errors

**Critical lesson:** Test set composition can be EXTREMELY misleading. Always check full prediction distribution before choosing threshold!

### Scenario 2: Well-Calibrated Model (Balanced Training or Class Weighting)

**Characteristics:**
- Model trained with class_weight='balanced' or scale_pos_weight
- Or trained on balanced/SMOTE data
- Predictions spread across FULL range (0.0-1.0)

**Recommended threshold: 0.5-0.7**

**Why lower threshold needed:**
```python
# Person genuinely at high risk
# Model correctly predicts: 0.75

exp.generate_counterfactuals(
    query_instance=high_risk_patient,
    stopping_threshold=0.9  # Target <0.10 - WILL FAIL!
)
# Needs 0.65 drop - impossible with realistic changes

# Better approach:
exp.generate_counterfactuals(
    query_instance=high_risk_patient,
    stopping_threshold=0.5  # Target <0.375 - achievable!
)
# Needs 0.375 drop - possible with multiple changes
```

**Warning:** These models often have inflated probabilities that make CF generation challenging!

### Scenario 3: Class-Weighted Model (Problematic - Consider Retraining)

**Characteristics:**
- Model trained with aggressive class_weight or scale_pos_weight
- Severely inflated probabilities
- Poor CF generation performance

**Recommended approach:**
- Threshold: **0.3-0.5** (very conservative)
- **Better: Retrain model without class weighting** for CF use case

**Expected results:**
- Query success: 20-80% (highly variable)
- Valid CFs: 10-30 per 9 queries
- May require extreme feature changes

### Scenario 4: Exploratory/Research

**Characteristics:**
- Understanding CF generation patterns
- Testing different configurations
- Not for production use

**Recommended approach:**
- Test multiple thresholds: **0.1, 0.5, 0.9**
- Compare results across thresholds
- Analyze trade-offs

**Focus:**
- How does threshold affect CF success?
- Trade-off between ambition and feasibility
- Interaction with other parameters (sparsity, proximity)

---

## 8. Interactions with Other Parameters

### Stopping Threshold + Sparsity

**Independent but complementary:**

- **Stopping threshold**: How much to reduce probability
- **Sparsity**: How many features to change

**Example:**
```yaml
stopping_threshold: 0.9  # Ambitious probability target
posthoc_sparsity_param: 0.1  # Prefer fewer feature changes
```

**Effect:**
- DiCE tries to reach <10% risk using as few features as possible
- May need to make LARGER changes to fewer features
- Trade-off: Sparsity vs. magnitude of changes

### Stopping Threshold + Proximity Weight

**Related objectives:**

- **Stopping threshold**: Defines valid CF (probability-based)
- **Proximity weight**: Encourages small changes (distance-based)

**Higher proximity weight:**
- Prefers smaller feature changes
- May conflict with ambitious threshold
- Can make high thresholds harder to achieve

**Lower proximity weight:**
- Allows larger feature changes
- Easier to reach ambitious thresholds
- May produce less realistic CFs

### Stopping Threshold + Diversity Weight

**Complementary:**

- **Stopping threshold**: All CFs must meet probability target
- **Diversity weight**: Encourages different solutions among valid CFs

**Effect:**
- Threshold filters what's valid
- Diversity spreads out the valid solutions
- Higher threshold = smaller valid region = less room for diversity

---

## 9. Debugging Threshold Issues

### Problem: No valid CFs found (0% success)

**Diagnosis:**
1. Check model probabilities on test set
2. Compare starting probabilities to target

**Solutions:**
- Lower the threshold (0.9 → 0.7 → 0.5)
- Check if model has inflated probabilities (class weighting?)
- Increase maxiterations (give DiCE more time)
- Reduce other constraints (permitted_range, features_to_vary)

### Problem: All CFs too similar (low diversity)

**Diagnosis:**
- Threshold may be too high
- Valid region is very small

**Solutions:**
- Lower threshold slightly (0.9 → 0.8)
- Increase diversity weight
- Check if posthoc_sparsity is too restrictive

### Problem: CFs are unrealistic (extreme changes)

**Diagnosis:**
- Threshold too ambitious for starting probability
- Model probabilities may be problematic

**Solutions:**
- Lower threshold
- Increase proximity weight
- Add tighter permitted_range constraints
- Check model calibration

---

## 10. Best Practices

### Do's

✓ **Check model's prediction distribution FIRST** before choosing threshold

```python
# Always do this first!
y_pred = model.predict_proba(X_test)[:, 1]
print(f"Max prediction: {y_pred.max():.3f}")
# Use decision rules from section 5.2
```

✓ **Start with appropriate threshold** based on model's max predictions
- Max <0.35: Start with 0.9
- Max 0.35-0.60: Start with 0.7
- Max >0.60: Start with 0.5

✓ **Match domain requirements** (clinical significance > optimization success)

✓ **Test multiple thresholds** when exploring

✓ **Document your choice** with rationale (include max predictions from model)

✓ **Monitor query success rate** (aim for >80%)

### Don'ts

✗ **Don't assume threshold depends on population** (it depends on MODEL's prediction distribution!)

✗ **Don't use same threshold for all models** (depends on training strategy and calibration)

✗ **Don't choose threshold without checking model's max predictions first**

✗ **Don't ignore failed queries** (understand why they fail)

✗ **Don't use very low thresholds** (<0.3) unless model has severely inflated probabilities

✗ **Don't forget formula: target_prob = original_prob × (1 - stopping_threshold)**

---

## 11. Summary

**Key Takeaways:**

1. **Model's MAXIMUM prediction determines threshold choice** (NOT median or mean!)
   - Models with max pred 0.95-0.98: Use threshold 0.5-0.7 ✓ (most common)
   - Models with max pred 0.30-0.60: Use threshold 0.7-0.8
   - Models with max pred <0.30: Use threshold 0.9 (rare, pre-filtered only)
   - **Always check MAX, not just percentiles!**

2. **Test set composition can be EXTREMELY misleading:**
   - CF test (9 individuals): max pred 0.20 → threshold 0.9 worked perfectly
   - Full test set: max pred 0.97 → threshold 0.9 fails for ~5%
   - Don't extrapolate from small or cherry-picked test sets!

3. **Absolute distance matters**, not relative change
   - Starting at 6% with threshold 0.9: Easy (need 0% reduction)
   - Starting at 48% with threshold 0.9: Hard (need 38% reduction)

4. **Real-world example validates this:**
   - Test set: 2/9 have CVD (mixed sick/healthy)
   - Model max prediction: 0.20 (conservative from imbalanced training)
   - Threshold 0.9 achieved 100% success
   - Proves that threshold depends on MODEL, not population

5. **Threshold interacts with other parameters**
   - Sparsity: How many features to change
   - Proximity: How much to change features
   - Diversity: Variety among valid CFs

**Decision flowchart for choosing threshold (CORRECTED):**
```yaml
# Step 1: Check model's max predictions on FULL representative test set
max_pred = model.predict_proba(X_test)[:, 1].max()
percentile_95 = np.percentile(model.predict_proba(X_test)[:, 1], 95)

# Step 2: Choose threshold based on MAX (not percentile!)
if max_pred < 0.35:
    stopping_threshold: 0.9  # Rare - only for pre-filtered low-risk
    # WARNING: Verify test set is representative!

elif max_pred < 0.60:
    stopping_threshold: 0.7  # Moderate range
    # Some compressed distributions

else:  # max_pred >= 0.60
    stopping_threshold: 0.5  # RECOMMENDED for most production systems
    # Handles full spectrum including genuinely high-risk users
    # Even if 95% of users are low-risk, that 5% needs working CFs!
```

**Critical lesson from real experiments (CORRECTED):**
- **CF test (misleading)**: 9 individuals, max pred 0.20 → threshold 0.9 = 100% success
- **Full test (reality)**: Same model, max pred 0.97 → threshold 0.9 would fail for high-risk
- **Conclusion**: Small test sets can be unrepresentative! Always check full distribution.
- **Proof**: Threshold depends on MODEL'S FULL RANGE, not sample statistics!

---

## 12. Further Reading

**Related documentation:**
- `Understanding_CF_Generation_and_Model_Compatibility.md` - Deep dive into probability inflation issues
- `DiCE_sparsity_overview.md` - How sparsity parameter works
- `DiCE_proximity_overview.md` - How proximity parameter works

**Key concepts:**
````
- Model probability calibration
- Class imbalance handling (class weighting effects)
- Feature budget and realistic change bounds
- Trade-offs between CF quality dimensions

---

**Document Version**: 1.0
**Date**: May 7, 2026
**Related Parameters**: sparsity, proximity, diversity, maxiterations
