# Analysis Scripts for Generation 2 Experiments

This directory contains Python scripts used to analyze experimental results and generate the findings documented in `docs/Generation_2_Experimental_Results_Analysis.md`.

## Scripts

### 01_analyze_base_vs_thresholds.py

Analyzes baseline RF and XGB models with different stopping thresholds.

**Input:** `cf_outputs/base_vs_thresholds/`
**Output:** `docs/Predictors_vs_threshold/analysis_scripts/output/base_vs_thresholds_summary.csv`

**Usage:**
```bash
cd /path/to/cf_bench
python docs/Predictors_vs_threshold/analysis_scripts/01_analyze_base_vs_thresholds.py
```

### 02_analyze_smote_and_optimized.py

Analyzes SMOTE-trained models and XGB models with optimized hyperparameters.

**Input:**
- `cf_outputs/SMOTE/base_predictors/`
- `cf_outputs/SMOTE/gridsearched_predictors/`
- `cf_outputs/xgb_optimized/`

**Output:** `docs/Predictors_vs_threshold/analysis_scripts/output/smote_xgb_optimized_summary.csv`

**Usage:**
```bash
cd /path/to/cf_bench
python docs/Predictors_vs_threshold/analysis_scripts/02_analyze_smote_and_optimized.py
```

### 03_comprehensive_comparison.py

Combines all results and generates rankings and recommendations.

**Input:**
- `docs/Predictors_vs_threshold/analysis_scripts/output/base_vs_thresholds_summary.csv`
- `docs/Predictors_vs_threshold/analysis_scripts/output/smote_xgb_optimized_summary.csv`

**Output:** `docs/Predictors_vs_threshold/analysis_scripts/output/all_experiments_combined.csv`

**Usage:**
```bash
cd /path/to/cf_bench
python docs/Predictors_vs_threshold/analysis_scripts/03_comprehensive_comparison.py
```

## Running All Analyses

To reproduce the complete analysis, run scripts in order:

```bash
cd /path/to/cf_bench
python docs/Predictors_vs_threshold/analysis_scripts/01_analyze_base_vs_thresholds.py
python docs/Predictors_vs_threshold/analysis_scripts/02_analyze_smote_and_optimized.py
python docs/Predictors_vs_threshold/analysis_scripts/03_comprehensive_comparison.py
```

Or use the provided convenience script:

```bash
cd /path/to/cf_bench
bash docs/Predictors_vs_threshold/analysis_scripts/run_all_analyses.sh
```

## Requirements

- Python 3.7+
- pandas
- pathlib (standard library)

Install dependencies:
```bash
pip install pandas
```

## Output Files

All output CSV files are saved to `docs/Predictors_vs_threshold/analysis_scripts/output/`:

- `base_vs_thresholds_summary.csv` - Baseline model results
- `smote_xgb_optimized_summary.csv` - SMOTE and optimized model results
- `all_experiments_combined.csv` - Combined dataset for all experiments

## Metrics Calculated

Each script calculates the following metrics per configuration:

- **n_queries**: Number of query instances tested
- **total_cfs_generated**: Total CFs attempted to generate
- **valid_cfs**: Number of valid counterfactuals
- **validity_rate**: Proportion of valid CFs among generated
- **query_success_rate**: Percentage of queries with at least 1 valid CF
- **avg_gower_distance**: Average Gower distance for valid CFs
- **avg_features_changed**: Average number of features changed in valid CFs
- **avg_gen_time_sec**: Average generation time per query

## Notes

- Scripts assume they are run from the project root directory (`cf_bench/`)
- Input data structure follows the experiment output format (folders with `annotated.csv` files)
- Output directory is created automatically if it doesn't exist
- Scripts are standalone and can be run independently if needed

## Date

Generated: May 7, 2026
Author: Analysis for Generation 2 Experiments
