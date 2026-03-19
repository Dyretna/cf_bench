# Counterfactual Transitions in Cardiovascular Disease (CVD) Predictions

## Project Overview

With the rapid growth of machine learning and explainable artificial intelligence (XAI) in healthcare, **counterfactual explanations** have emerged as a powerful tool for making predictive models actionable at the individual level.

This project investigates **counterfactual transitions in cardiovascular disease (CVD) risk predictions**, focusing on how small, feasible changes in modifiable risk factors can lead to safer prediction outcomes. Rather than population-level recommendations, this work supports **personalized prevention**, empowering clinicians, policymakers, and individuals to understand which specific interventions are most impactful for reducing cardiovascular risk.

Counterfactual explanations identify the *minimal feasible modification* required in a patient’s risk profile to change a predicted outcome (e.g., from high risk to low risk). By linking modifiable features—such as smoking behavior, BMI, blood pressure, education, or income proxies—to changes in predicted CVD risk, this project bridges the gap between statistical prediction and actionable public health or clinical decision-making.


## Research Questions

This project addresses the following research questions:

1. **How do the characteristics of counterfactual explanations vary between different counterfactual generation methods** when applied to cardiovascular risk prediction models?
2. **Which lifestyle changes most effectively reduce an individual’s predicted CVD risk**, under constraints of minimality and feasibility?
3. **Which combinations of machine learning models and counterfactual algorithms provide the most accurate and meaningful predictions**, across:
   - Lifestyle features  
   - Social factors  
   - Demographic characteristics  

In addition, generated counterfactual recommendations can be **audited for fairness and feasibility**, making the framework suitable for responsible AI in health research.


## Methods and Tools

- **Predictive model:** Random Forest classifier for CVD risk prediction  
- **Counterfactual framework:** DiCE (Diverse Counterfactual Explanations)  
- **Approach:** Model-agnostic counterfactual generation  
- **Data source:** European Social Survey (ESS) derived dataset (cleaned and feature-engineered)

---

## Project Structure
```text
cvd-counterfactuals/
│
├── data/
│   ├── ess.csv                     # Raw ESS data
│   ├── ess_clean_full.csv          # Cleaned dataset
│   └── ess_model_ready.csv         # Final dataset used by all models
│
├── cleaning/
│   └── clean_data.py               # Data cleaning and preprocessing
│
├── models/
│   └── train_rf.py                 # Random Forest model training
│
├── counterfactuals/
│   └── dice_cf.py                  # DiCE-based counterfactual generation
│
├── outputs/
│   ├── metrics/                    # Model evaluation results
│   └── counterfactuals/            # Generated counterfactual explanations
│
└── requirements.txt                # Python dependencies


> **Important:**  
> All models and counterfactual analyses are based on  
> `data/ess_model_ready.csv`.


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Roozhinkh/CF-Transitions-in-CVD-Predictions.git
cd CF-Transitions-in-CVD-Predictions

2. (Recommended) Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Install dependencies:
pip install -r requirements.txt


How to Run the Project
1. Clean and preprocess the data
python cleaning/clean_data.py
This generates cleaned datasets used for modeling and counterfactual analysis.

2. Train the Random Forest model
python models/train_rf.py
This step trains the CVD risk prediction model and stores evaluation metrics and trained artifacts.

3. Generate counterfactual explanations
python counterfactuals/dice_cf.py
This script generates counterfactual explanations using DiCE and saves the results to the outputs/counterfactuals/ directory.

Outputs:
- Model performance metrics: stored in outputs/metrics/
- Counterfactual explanations: stored in outputs/counterfactuals/
- Query instances and generated counterfactuals: available as CSV files for further analysis

Contribution Guidelines
Contributions are welcome!
1. Fork the repository
2. Create a new branch:
git checkout -b feature/your-feature-name

3. Commit your changes:
git commit -m "Add: description of your contribution"

4. Push to your fork and open a Pull Request

Notes on Ethics and Fairness
This project emphasizes interpretability, feasibility, and fairness in counterfactual explanations. Recommendations should be interpreted as decision-support tools, not clinical diagnoses. Care must be taken when using social or demographic variables in healthcare decision-making.
