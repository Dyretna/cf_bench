# Optimizing Predictors

This folder contains notebooks and results from hyperparameter optimization experiments for Random Forest and XGBoost models used in counterfactual generation benchmarks.

## Overview

The notebooks perform RandomizedSearchCV to find optimal hyperparameters for classifiers trained on cardiovascular disease (CVD) prediction tasks. The data is imbalanced, so class weighting strategies are explored extensively.

## Files

### Notebooks

- **[rf_gridsearch.ipynb](rf_gridsearch.ipynb)**: Random Forest hyperparameter search using RandomizedSearchCV with 200 iterations. Tests various combinations of tree depth, estimators, sampling strategies, and class weighting methods.

- **[rf_save_optimized.ipynb](rf_save_optimized.ipynb)**: Loads gridsearch results, selects the best Random Forest model configuration based on mean cross-validation scores, trains the final model, and saves it as `rf_optimized_hltprhc.pkl`.

- **[xgb_gridsearch.ipynb](xgb_gridsearch.ipynb)**: XGBoost hyperparameter search with two phases - an initial broad search (400 iterations) and a refined search (300 iterations) focusing on promising parameter ranges. Explores learning rate, tree depth, regularization, and `scale_pos_weight` for handling class imbalance.

- **[xgb_save_optimized.ipynb](xgb_save_optimized.ipynb)**: Analyzes XGBoost gridsearch results, manually selects optimal parameters based on patterns in top-performing configurations, trains the final model, and saves it as `xgb_optimized_hltprhc.pkl`.

### Results Files

- **rf_gridsearch_runs.csv**: Aggregated results from Random Forest hyperparameter searches, including cross-validation scores and parameter combinations.

- **xgb_gridsearch_runs.csv**: Aggregated results from XGBoost hyperparameter searches, with detailed metrics for each configuration tested.

## Key Findings

### Class Imbalance Handling

Both model families show the classic precision-recall tradeoff when using class weighting:
- **Unweighted models** favor the majority class, resulting in high false-negative rates
- **Class-weighted models** (`balanced`, `balanced_subsample`, or `scale_pos_weight`) shift the decision boundary toward the minority class, reducing false negatives but increasing false positives

### Performance Plateau

Both Random Forest and XGBoost reached performance ceilings across extensive hyperparameter searches:
- **Random Forest**: Mean macro-recall values varied only marginally across 200 iterations
- **XGBoost**: Convergence around similar performance metrics even after refined searches

This plateau indicates that further hyperparameter tuning is unlikely to yield substantial improvements. Better results would require:
- Feature engineering
- Decision threshold adjustment
- Probability calibration
- Alternative modeling approaches

### Optimal Parameters

**Random Forest** (hltprhc target):
- n_estimators: 450
- max_depth: 6
- min_samples_split: 2
- min_samples_leaf: 3
- max_features: "sqrt"
- class_weight: "balanced_subsample"

**XGBoost** (hltprhc target):
- n_estimators: 450
- max_depth: 4
- learning_rate: 0.05
- subsample: 0.75
- colsample_bytree: 0.70
- min_child_weight: 3
- gamma: 1.2
- scale_pos_weight: 10
- reg_lambda: 2.0
- reg_alpha: 1.0

## Usage

The notebooks expect environment variables defined in a `.env` file:
- `DATA_DIR`: Path to training/test data (eta.csv, ete.csv)
- `MODELS_DIR`: Path where optimized models should be saved

All notebooks use the `cf_bench.config.SystemConfig` for consistent feature selection and target specification.

## Subfolder

- **[undersampling_strategy/](undersampling_strategy/)**: Experiments comparing model performance and counterfactual generation quality between balanced (undersampled) and unbalanced training datasets.
