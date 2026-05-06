# Runs on SMOTE...
- high stopping_threshold = 0.9
- mid stopping_threshold = 0.5
- low stopping_threshold = 0.1

- base = based on model trained with SMOTE, but without gridsearch parameter tuning
- gs = based on model trained with SMOTE, with gridsearch parameter tuning

# results overview:

### Random Forest
- **base_rf_SMOTE_highthres**: produces good results, All valid and expected
- **base_rf_SMOTE_midthres**:  produces good results, one observation fails (cannot find valid)
- **base_rf_SMOTE_lowthres**:  [ERROR] No counterfactuals found for any of the query points!

- **gs_rf_SMOTE_highthres**: [ERROR] No counterfactuals found for any of the query points!
- **gs_rf_SMOTE_midthres**: Produces poor results, 5 invalid.
- **gs_rf_SMOTE_lowthres**: [ERROR] No counterfactuals found for any of the query points!


### XGBooost
- not done yet... upcoming...

for more info, see notebooks, spreadsheets, etc...



# comment
- highthreshold is best for simple RF, then successively worse...
