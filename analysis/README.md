# Analysis CLI

Command-line tool for analyzing counterfactual experiment results.

## Usage

```bash
# Analyze all experiments
python -m analysis.cli --all --output results.csv

# Analyze specific directory
python -m analysis.cli --dir predictors_vs_threshold/baseline

# Analyze with pattern matching
python -m analysis.cli --pattern "**/XGBoost**/annotated.csv"

# Include constraint parameters (sparsity, locked features)
python -m analysis.cli --all --include-config

# Quiet mode (no console output)
python -m analysis.cli --all --output results.csv --quiet
```

## Options

- `--all` - Analyze all experiments in cf_outputs
- `--dir <path>` - Analyze experiments in specific subdirectory
- `--pattern <glob>` - Match files using glob pattern
- `--output <file>` - Save results to CSV file
- `--include-config` - Include sparsity and locking columns
- `--quiet` - Suppress console output
- `--base-path <path>` - Custom cf_outputs location (default: CF_OUTPUTS env var or ../cf_outputs)

## Output

Generates a comparison table with metrics:
- **Configuration**: explainer type, model type, parameters
- **Validity metrics**: validity %, solved %
- **Feature changes**: avg Nchanged, Gower distance
- **Risk metrics**: risk reduction, avg risk before/after
- **Performance**: generation time per patient
- **Feature analysis**: most frequently changed features

## Requirements

- Python 3.11+
- pandas, numpy
- python-dotenv
