# Batch Config Runner Usage

## Overview

The `batch_config_runner.py` script allows you to batch run all YAML configuration files in a directory tree. This is essential for running large experiment sets (like `predictors_vs_threshold` with 27 configs) without manual intervention.

## Basic Usage

```bash
# Run all configs in a directory
python scripts/batch_config_runner.py configs/predictors_vs_threshold

# Run only a subdirectory
python scripts/batch_config_runner.py configs/predictors_vs_threshold/baseline

# Preview what will run (dry-run)
python scripts/batch_config_runner.py configs/predictors_vs_threshold --dry-run

# Run with debug output
python scripts/batch_config_runner.py configs/predictors_vs_threshold --debug
```

## Default Behavior: Continue on Error

The script is designed for **batch processing** and continues running even if individual configs fail. This means:

- ✅ Run large experiment sets overnight without manual intervention
- ✅ Get results from all successful configs even if some fail
- ✅ Review failures in summary report afterward

If you need to stop on first failure:
```bash
python scripts/batch_config_runner.py configs/predictors_vs_threshold --stop-on-error
```

## Output Structure

**Understanding where things go is critical:**

### 1. Individual Experiment Results
Each YAML config specifies its own `output_dir` where experiment results are saved:
```yaml
# In rf_highthres.yaml
output_dir: "cf_outputs/base_vs_thresholds/baseline/"
```
Results go there → `cf_outputs/base_vs_thresholds/baseline/base_rf_highthres_2026-05-06/`

### 2. Batch Run Logs (Separate Location)
The batch script creates logs in a centralized location with descriptive naming:
```
cf_outputs/batch_runs/
  predictors_vs_threshold_2026-05-11_09-04-36/
    batch_run.log         # Complete output from all runs
    summary.txt           # Success/failure summary

  predictors_vs_threshold_baseline_2026-05-11_10-15-22/
    batch_run.log
    summary.txt
```

**Why separate?**
- Batch logs document what was run and when
- Experiment results go where the configs say (may be shared across runs)
- Easy to find "what happened in that batch run on May 11"
- Folder names clearly show which config directory was processed

## Log Files

### `batch_run.log`
Complete timestamped output including:
- All stdout/stderr from each config
- Progress markers
- Error messages with full details

### `summary.txt`
Clean overview showing:
- Success/failure counts
- List of failed configs with specific error reasons
- List of all successful configs
- Paths to log files

### Example Summary (with failures)

```
================================================================================
BATCH RUN SUMMARY
================================================================================
Timestamp: 2026-05-11_09-04-36
Config Directory: configs/predictors_vs_threshold

Completed: 24/27 successful

Failed configs (3):
  ✗ SMOTE/gs_xgb_SMOTE_highthres.yaml
    Reason: Exit code 1
  ✗ weighted/optXGB_high_thres.yaml
    Reason: Exit code 1
  ✗ optimized_XGBoost/xgboost_midthres.yaml
    Reason: FileNotFoundError: model not found

Successful configs:
  ✓ SMOTE/base_rf_SMOTE_highthres.yaml
  ✓ SMOTE/base_rf_SMOTE_lowthres.yaml
  ✓ SMOTE/base_rf_SMOTE_midthres.yaml
  ... (21 more)
================================================================================
```

## Typical Workflow

### 1. Start Batch Run
```bash
python scripts/batch_config_runner.py configs/predictors_vs_threshold
```
Output shows: `Logs will be saved to: cf_outputs/batch_runs/predictors_vs_threshold_2026-05-11_09-04-36`

### 2. Let It Run (Overnight)
- Continues even if some configs fail
- All output captured to log files

### 3. Review Results (Next Day)
```bash
# Quick check
cat cf_outputs/batch_runs/predictors_vs_threshold_2026-05-11_09-04-36/summary.txt

# Detailed investigation of failures
less cf_outputs/batch_runs/predictors_vs_threshold_2026-05-11_09-04-36/batch_run.log
```

### 4. Access Experiment Results
Individual experiment outputs are in their configured locations:
```bash
ls cf_outputs/base_vs_thresholds/baseline/
ls cf_outputs/base_vs_thresholds/SMOTE/base_predictors/
```

## Advanced Options

### Stop on First Error
Not recommended for batch runs, but available:
```bash
python scripts/batch_config_runner.py configs/predictors_vs_threshold --stop-on-error
```

### Debug Mode
Enables detailed debug output from each run:
```bash
python scripts/batch_config_runner.py configs/predictors_vs_threshold --debug
```

## Tips

- **Always use dry-run first** to preview what will be executed
- **Check summary.txt** before diving into full logs
- **Folder names include config directory** so you know what was run
- **Timestamp in folder name** lets you track when runs happened
- **Failed configs show error reasons** so you know what to fix
