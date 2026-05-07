# Runs on SMOTE...
- high stopping_threshold = 0.9
- mid stopping_threshold = 0.5
- low stopping_threshold = 0.1

- base = based on model trained with SMOTE, but without gridsearch parameter tuning
- gs = based on model trained with SMOTE, with gridsearch parameter tuning

# results overview:

### Random Forest
- **base_rf_SMOTE_highthres**:  1 invalid
- **base_rf_SMOTE_midthres**:   1 invalid
- **base_rf_SMOTE_lowthres**:   [ERROR] No counterfactuals found for any of the query points!

- **gs_rf_SMOTE_highthres**:    5 invalid.
- **gs_rf_SMOTE_midthres**:     7 invalid.
- **gs_rf_SMOTE_lowthres**:     [ERROR] No counterfactuals found for any of the query points!


### XGBooost
- **base_xgb_SMOTE_highthres**: 2 invalid
- **base_xgb_SMOTE_midthres**:  2 invalid
- **base_xgb_SMOTE_lowthres**:  [ERROR] No counterfactuals found for any of the query points!

- **gs_xgb_SMOTE_highthres**:   [ERROR] No counterfactuals found for any of the query points!
- **gs_xgb_SMOTE_midthres**:    3 invalid
- **gs_xgb_SMOTE_lowthres**:    [ERROR] No counterfactuals found for any of the query points!

# comment
- highthreshold is best for simple RF, then successively worse...
- there are random variations between runs.

for more info, see notebooks, spreadsheets, etc...
