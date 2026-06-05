# Counterfactual Explanations in Python using DiCE

This repository contains the full implementation, experiments and analysis from my thesis project on counterfactual explanations (CFs) applied to public health survey data. The goal of the project is to evaluate how different machine learning models and DiCE explainers behave when generating counterfactuals, and how model calibration, probability structure and parameter settings (such as `stopping_threshold`) influence the validity and feasibility of the generated CFs.

The project uses data from the European Social Survey (ESS) and focuses on predicting cardiovascular disease based on lifestyle variables. Two prediction models are explored — **Random Forest** and **XGBoost** — together with DiCE’s **random-search** and **genetic** explainers. The experiments are organized into three phases:

- Baseline models, explainers, constraints
- Model optimization via SMOTE and class weighting
- Threshold‑sensitivity experiments to study how CF generation interacts with model calibration


---

## Repository Structure
```
.
├── cf_outputs/         # Generated counterfactuals
├── configs/            # YAML configuration files for pipeline runs
├── data/               # dataset directory
├── notebooks/          # Jupyter notebooks for training ML-models, cf-experiments and analysis
├── src/                # Python package source code
│   └── cf_bench/
├── models/             # Saved predictor models
├── pyproject.toml      # Package definition
└── README.md
```

---


## Installation
Install the project as a local package:

```bash
pip install -e .
```

after installation, modules can be imported directly

## Configuration (YAML)
The pipeline is configured using YAML files stored in the configs/ directory.
These files define paths, some of the settings and parameters.


## Environment Variables (`.env`)
This repository uses a `.env` file to store local paths dynamically.
This makes it easier to import directories consistently in notebooks.

Create a `.env` file in the project root and define any paths you need, for example:
```
DATA_DIR=/path/to/data
MODELS_DIR=/path/to/models
CF_OUTPUTS=/path/to/counterfactuals
```

## Usage
1. Create a .env file with your local paths
2. Select or edit a YAML config in configs/
3. Generated counterfactuals (CSV files, metrics, metadata) will be saved in the directory specified in the config.


### Running the Pipeline
Basic usage:
```bash
python -m cf_bench.cli --config configs/rf_hltprhc_cfcheck.yaml
```

With debug output (shows detailed dtype info, query details, etc.):
```bash
python -m cf_bench.cli --config configs/xgboost_hltprhc_cfcheck.yaml --debug
```

### Logging Levels

- **Normal mode** (default): Shows INFO level messages - pipeline progress, major steps
- **Debug mode** (`--debug` flag): Shows DEBUG level messages - detailed dtype conversions, query instance details, SanitizedModel operations

Examples of output:
```
# Normal mode
INFO: Starting counterfactual pipeline for target: hltprhc
INFO: Explainer profile: genetic
INFO: Generating counterfactuals for 50 instances...
INFO: Exporting results...
INFO: Output directory: cf_outputs/genetic_hltprhc_2026-04-17

# Debug mode
2026-04-17 14:23:45 - cf_bench.dice_batch_runner - DEBUG - model_input_df dtypes (for SanitizedModel):
2026-04-17 14:23:45 - cf_bench.dice_batch_runner - DEBUG -   bmi: float64
2026-04-17 14:23:45 - cf_bench.dice_batch_runner - DEBUG -   etfruit: object
...
```

---
