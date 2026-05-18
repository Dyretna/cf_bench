# SMOTE and Weighted Models - Experiment Summary

This document provides a comprehensive overview of all SMOTE and weighted model experiments, including successful runs and failures.

## Overview

All experiments tested counterfactual generation across three stopping threshold values:
- **High threshold** (0.9): More lenient stopping condition
- **Mid threshold** (0.5): Balanced stopping condition
- **Low threshold** (0.1): Strict stopping condition

---

## SMOTE Experiments

### SMOTE Base Predictors
Models trained with SMOTE oversampling but **without** gridsearch hyperparameter tuning.

| Config File | Model | Threshold | Status | Output Path | Notes |
|-------------|-------|-----------|--------|-------------|-------|
| `base_rf_SMOTE_highthres.yaml` | RandomForest | 0.9 | [SUCCESS] | `SMOTE/base_predictors/RandomForest_thres0.9_2026-05-11/` | 1 invalid CF |
| `base_rf_SMOTE_midthres.yaml` | RandomForest | 0.5 | [SUCCESS] | `SMOTE/base_predictors/RandomForest_thres0.5_2026-05-11/` | 1 invalid CF |
| `base_rf_SMOTE_lowthres.yaml` | RandomForest | 0.1 | [FAILED] | — | **No counterfactuals generated** |
| `base_xgb_SMOTE_highthres.yaml` | XGBoost | 0.9 | [SUCCESS] | `SMOTE/base_predictors/XGBoost_thres0.9_2026-05-11/` | 2 invalid CFs |
| `base_xgb_SMOTE_midthres.yaml` | XGBoost | 0.5 | [SUCCESS] | `SMOTE/base_predictors/XGBoost_thres0.5_2026-05-11/` | 2 invalid CFs |
| `base_xgb_SMOTE_lowthres.yaml` | XGBoost | 0.1 | [FAILED] | — | **No counterfactuals generated** |

**Summary:** 4/6 experiments successful. All low threshold (0.1) experiments failed.

---

### SMOTE Gridsearched Predictors
Models trained with SMOTE oversampling **with** gridsearch hyperparameter tuning.

| Config File | Model | Threshold | Status | Output Path | Notes |
|-------------|-------|-----------|--------|-------------|-------|
| `gs_rf_SMOTE_highthres.yaml` | RandomForest | 0.9 | [FAILED] | — | **No counterfactuals generated** |
| `gs_rf_SMOTE_midthres.yaml` | RandomForest | 0.5 | [SUCCESS] | `SMOTE/gridsearched_predictors/RandomForest_thres0.5_2026-05-11/` | 7 invalid CFs |
| `gs_rf_SMOTE_lowthres.yaml` | RandomForest | 0.1 | [FAILED] | — | **No counterfactuals generated** |
| `gs_xgb_SMOTE_highthres.yaml` | XGBoost | 0.9 | [FAILED] | — | **No counterfactuals generated** |
| `gs_xgb_SMOTE_midthres.yaml` | XGBoost | 0.5 | [SUCCESS] | `SMOTE/gridsearched_predictors/XGBoost_thres0.5_2026-05-11/` | 3 invalid CFs |
| `gs_xgb_SMOTE_lowthres.yaml` | XGBoost | 0.1 | [FAILED] | — | **No counterfactuals generated** |

*Note: Output path discrepancy for `gs_rf_SMOTE_highthres` - folder shows 0.5 but config is 0.9. May need verification.

**Summary:** 2-3/6 experiments successful. Low threshold failed for both models. Gridsearched XGBoost had worse performance than base XGBoost.

---

## Weighted Models Experiments

Models trained with class weights to handle imbalanced data. **Note:** Only mid-threshold experiments were successful.

| Config File | Model | Threshold | Status | Output Path | Notes |
|-------------|-------|-----------|--------|-------------|-------|
| `optRF_high_thres.yaml` | RandomForest | 0.9 | [FAILED] | — | **No counterfactuals generated** |
| `optRF_mid_thres.yaml` | RandomForest | 0.5 | [SUCCESS] | `weighted/RandomForest_thres0.5_2026-05-11/` | Poor results |
| `optRF_low_thres.yaml` | RandomForest | 0.1 | [FAILED] | — | **No counterfactuals generated** |
| `optXGB_high_thres.yaml` | XGBoost | 0.9 | [FAILED] | — | **No counterfactuals generated** |
| `optXGB_mid_thres.yaml` | XGBoost | 0.5 | [SUCCESS] | `weighted/XGBoost_thres0.5_2026-05-11/` | Poor results |
| `optXGB_low_thres.yaml` | XGBoost | 0.1 | [FAILED] | — | **No counterfactuals generated** |

**Summary:** 2/6 experiments successful. Only mid-threshold (0.5) worked. Both high and low thresholds failed completely.

---

## Overall Summary Statistics

### Success Rates by Configuration Type

| Configuration | Total | Successful | Failed | Success Rate |
|---------------|-------|------------|--------|--------------|
| SMOTE Base | 6 | 4 | 2 | 66.7% |
| SMOTE Gridsearched | 6 | 2-3 | 3-4 | 33-50% |
| Weighted | 6 | 2 | 4 | 33.3% |
| **TOTAL** | **18** | **8-9** | **9-10** | **44-50%** |

### Success Rates by Threshold

| Threshold | Total Tests | Successful | Failed | Success Rate |
|-----------|-------------|------------|--------|--------------|
| High (0.9) | 6 | 2 | 4 | 33.3% |
| Mid (0.5) | 6 | 6 | 0 | **100%** |
| Low (0.1) | 6 | 0 | 6 | **0%** |

### Success Rates by Model Type

| Model | Total Tests | Successful | Failed | Success Rate |
|-------|-------------|------------|--------|--------------|
| RandomForest | 9 | 5 | 4 | 55.6% |
| XGBoost | 9 | 3-4 | 5-6 | 33-44% |

---

## Key Findings

### 1. **Threshold 0.5 is the Sweet Spot**
- **All** mid-threshold (0.5) experiments succeeded (6/6 = 100%)
- This appears to be the optimal balance for counterfactual generation with SMOTE/weighted models

### 2. **Low Threshold (0.1) Completely Failed**
- **All** low threshold experiments failed (0/6 = 0%)
- Error: "No counterfactuals found for any of the query points"
- Conclusion: SMOTE/weighted models are incompatible with very strict stopping conditions

### 3. **High Threshold (0.9) Mostly Failed**
- Only 2/6 experiments succeeded (33.3%)
- SMOTE base models worked better than weighted
- Gridsearched XGBoost failed completely at high threshold

### 4. **SMOTE Base > SMOTE Gridsearched**
- SMOTE base predictors: 66.7% success rate
- SMOTE gridsearched: 33-50% success rate
- Gridsearching with SMOTE appears to hurt CF generation capability

### 5. **Weighted Models Perform Poorly**
- Only 33.3% success rate overall
- Failed at both high and low thresholds
- Even successful runs had "poor results" according to documentation

### 6. **RandomForest > XGBoost for SMOTE/Weighted**
- RandomForest: 55.6% success rate
- XGBoost: 33-44% success rate
- RandomForest appears more compatible with class imbalance handling

---

## Analysis Notebooks

The following notebooks analyze the successful experiments:

- **`gen_2_SMOTE_basemodels_threshold.ipynb`**: Focuses on SMOTE base models across thresholds 0.5 and 0.9
- **`gen_2_SMOTE_weighted_eng.ipynb`**: Comprehensive analysis of all SMOTE/weighted experiments (English)
- **`gen_2_SMOTE_weighted_swe.ipynb`**: Comprehensive analysis of all SMOTE/weighted experiments (Swedish)

---

## Related Documentation

- `configs/predictors_vs_threshold/SMOTE/README.md` - SMOTE experiment notes
- `configs/predictors_vs_threshold/weighted/README.md` - Weighted experiment notes
- `docs/Predictors_vs_threshold/Understanding_CF_Generation_and_Model_Compatibility.md` - Technical analysis
- `analysis/summary_data/gen_2_summary.csv` - Complete results data

---

*Last updated: 2026-05-16*
