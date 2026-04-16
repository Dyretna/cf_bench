# Counterfactual Transitions in Cardiovascular Disease (CVD) Predictions

## Project Overview
With the rapid growth of machine learning and explainable artificial intelligence (XAI) in healthcare, **counterfactual explanations** have emerged as a powerful tool for making predictive models actionable at the individual level.

This project investigates **counterfactual transitions in cardiovascular disease (CVD) risk predictions**, focusing on how small, feasible changes in modifiable risk factors can lead to safer prediction outcomes. Rather than population-level recommendations, this work supports **personalized prevention**, empowering clinicians, policymakers, and individuals to understand which specific interventions are most impactful for reducing cardiovascular risk.

Counterfactual explanations identify the *minimal feasible modification* required in a patient’s risk profile to change a predicted outcome (e.g., from high risk to low risk). By linking modifiable features—such as smoking behavior, BMI, blood pressure, education, or income proxies—to changes in predicted CVD risk, this project bridges the gap between statistical prediction and actionable public health or clinical decision-making.


## Project Goals - Experimentations for testing
Which combinations of:
- machine learning models and
- counterfactual algorithms

provide the most accurate and meaningful predictions across lifestyle factors?


## Methods and Tools
- **Data source:** European Social Survey (ESS) derived dataset (cleaned and feature-engineered)
- **Predictive models - for CVD risk prediction:**
      - Random Forest classifier
      - XGboost
      - tensorflow nn-models

- **Counterfactual framework:** DiCE (Diverse Counterfactual Explanations)
- **Approach:** Model-agnostic counterfactual generation

---
