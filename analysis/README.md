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
- **Validity metrics**:
  - `validity_%` - CFs that meet risk target
  - `solved_%` - patients with at least one valid CF
  - `actionable_%` - CFs respecting directional constraints (feasibility)
- **Feature changes**: avg Nchanged, Gower distance
- **Risk metrics**: risk reduction, avg risk before/after
- **Performance**: generation time per patient
- **Feature analysis**: most frequently changed features

### Actionability (Feasibility)

The `actionable_%` metric measures whether CFs respect directional constraints:
- **Should increase** (healthier when higher): cgtsmok, alcfreq, dosprt
- **Should decrease** (healthier when lower): bmi, etfruit, eatveg, slprl, paccnois

A CF is **actionable** if it only suggests changes in the beneficial direction. This aligns with DiCE's concept of feasibility/actionability - whether the suggested changes are realistic and make sense for improving health outcomes.

## Requirements

- Python 3.11+
- pandas, numpy
- python-dotenv
