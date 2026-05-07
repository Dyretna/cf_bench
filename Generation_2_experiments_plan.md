## Plan for Generation 2 – Experimental Design

### 1. Optimize the prediction models
- Address class imbalance (e.g., class_weight or resampling).
- Tune hyperparameters for both Random Forest and XGBoost.
- SMOTE training on RF and XGB (also with hyperparameter-tuning)
- Select two final candidate models (RF + XGB) based on stability and performance.

### 2. Focused test: model × stopping_threshold
- Explainer: Genetic (fixed).
- Constraints: Enabled (fixed).
- Factors varied:
  - base Models: RF vs XGBoost
  - SMOTE trained models: RF vs XGBoost
  - stopping_threshold: high vs mid vs low

- Purpose:
  - Confirm the trade-off between validity and solution diversity.
  - Check whether the patterns from Generation 1 remain with improved models.
  - Identify a model and a reasonable default threshold for the rest of the study.

### 3. Investigate sparsity (fine-tuning)
- Run a smaller test comparing:
  - standard sparsity vs high sparsity
  - using the best-performing model + threshold combination.
- Purpose:
  - Assess whether sparsity meaningfully affects plausibility or diversity.
  - If the effect is small, keep sparsity at a moderate level in the main experiments.

### 4. Final configuration
- Select:
  - the best model
  - the optimal stopping_threshold
  - a reasonable sparsity level
- Use this configuration for the final counterfactual generation and evaluation.
