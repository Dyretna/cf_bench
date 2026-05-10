"""
experiment summary - functional approach with dataclasses.

This module follows following design:
- Dataclass for experiment summaries (one row per experiment)
- Simple functions to compute metrics
- No complex class hierarchies
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config_parser import ConfigParser, find_config_file

# Feature columns
FEATURE_COLUMNS = [
    "etfruit",
    "eatveg",
    "cgtsmok",
    "alcfreq",
    "slprl",
    "paccnois",
    "bmi",
    "dosprt",
]

# Directional constraints for actionability check
# Features that should increase for better health (higher is better)
SHOULD_INCREASE = ["cgtsmok", "alcfreq", "dosprt"]
# Features that should decrease for better health (lower is better)
SHOULD_DECREASE = ["bmi", "etfruit", "eatveg", "slprl", "paccnois"]


@dataclass
class ExperimentSummary:
    """One row in the final comparison table."""

    # Identification
    experiment: str

    # Configuration
    explainer_type: str
    ml_model_type: str
    use_permitted_range: Optional[bool]
    total_cfs_requested: int
    maxiterations: int
    stopping_threshold: float

    # Data statistics
    n_patients: int
    total_cfs: int
    valid_cfs: int

    # Validity metrics
    validity_pct: float
    solved_pct: float
    actionable_pct: Optional[float]

    # Feature change metrics
    avg_nchanged: Optional[float]
    avg_nchanged_all: Optional[float]
    avg_gower_valid: Optional[float]

    # Risk metrics
    avg_risk_before_pct: Optional[float]
    avg_risk_after_pct: Optional[float]
    min_risk_after_pct: Optional[float]
    risk_reduction_pct: Optional[float]

    # Performance
    total_gen_time_sec: float
    avg_gen_time_sec: float

    # Feature analysis
    top_features: str

    # Optional constraints
    sparsity_param: Optional[float]
    locked_features: Optional[str]

    # Reference
    csv_path: str


def summarize_experiment(
    csv_path: Path, base_path: Path = None, include_constraints: bool = False
) -> ExperimentSummary:
    """
    Load one experiment and compute all metrics.

    Args:
        csv_path: Path to annotated.csv file
        base_path: Base path (cf_outputs) for computing relative paths
        include_constraints: Whether to include sparsity and locking columns

    Returns:
        ExperimentSummary with all computed metrics
    """
    # 1. Load annotated.csv
    df = pd.read_csv(csv_path)
    experiment_dir = csv_path.parent
    experiment_name = experiment_dir.name

    # Compute relative path from base_path (cf_outputs directory)
    if base_path:
        try:
            # Get path relative to cf_outputs (exclude the annotated.csv filename)
            rel_path = experiment_dir.relative_to(base_path)
            path_str = str(rel_path)
        except ValueError:
            # Fallback if path is not relative
            path_str = str(experiment_dir)
    else:
        path_str = str(experiment_dir)

    # 2. Extract config using ConfigParser
    config_path = find_config_file(experiment_dir)
    config = None
    if config_path:
        parser = ConfigParser(config_path)
        config = parser.parse()

    # 3. Split data
    original_rows = df[df["cf_id"] == "original"]
    cf_rows = df[df["cf_id"] != "original"]
    valid_cf_rows = cf_rows[cf_rows["valid"]]

    # 4. Compute metrics
    n_patients = len(original_rows)
    total_cfs = len(cf_rows)
    valid_cfs = len(valid_cf_rows)

    # Validity metrics
    validity_pct = (valid_cfs / total_cfs * 100) if total_cfs > 0 else 0.0
    solved_count = sum(
        cf_rows[cf_rows["query_index"] == qid]["valid"].any()
        for qid in original_rows["query_index"]
    )
    solved_pct = (solved_count / n_patients * 100) if n_patients > 0 else 0.0

    # Actionability metric (directional constraints)
    actionable_pct = _compute_actionable_pct(df, original_rows, cf_rows)

    # Feature change metrics
    avg_nchanged = _compute_avg_nchanged(valid_cf_rows)
    avg_nchanged_all = _compute_avg_nchanged(cf_rows)
    avg_gower_valid = _compute_avg_gower(valid_cf_rows)

    # Risk metrics
    avg_risk_before_pct = _compute_avg_risk_before(original_rows)
    avg_risk_after_pct = _compute_avg_risk_after(valid_cf_rows)
    min_risk_after_pct = _compute_min_risk_after(valid_cf_rows)
    risk_reduction_pct = _compute_risk_reduction(original_rows, valid_cf_rows)

    # Performance
    total_gen_time_sec = original_rows["cf_gen_time_sec"].sum()
    avg_gen_time_sec = original_rows["cf_gen_time_sec"].mean()

    # Feature analysis
    top_features = _compute_top_features(valid_cf_rows)

    # Config parameters (with defaults)
    explainer_type = config.explainer_profile if config else "unknown"
    ml_model_type = config.ml_model_type if config else "unknown"
    use_permitted_range = config.use_permitted_range if config else None
    total_cfs_requested = config.total_cfs if config else total_cfs
    maxiterations = config.maxiterations if config else 1000
    stopping_threshold = config.stopping_threshold if config else 0.5

    # Optional constraints
    sparsity_param = None
    locked_features = None
    if include_constraints and config:
        sparsity_param = config.posthoc_sparsity_param
        locked_features = (
            ", ".join(config.locked_features) if config.locked_features else None
        )

    return ExperimentSummary(
        experiment=experiment_name,
        explainer_type=explainer_type,
        ml_model_type=ml_model_type,
        use_permitted_range=use_permitted_range,
        total_cfs_requested=total_cfs_requested,
        maxiterations=maxiterations,
        stopping_threshold=stopping_threshold,
        n_patients=n_patients,
        total_cfs=total_cfs,
        valid_cfs=valid_cfs,
        validity_pct=validity_pct,
        solved_pct=solved_pct,
        actionable_pct=actionable_pct,
        avg_nchanged=avg_nchanged,
        avg_nchanged_all=avg_nchanged_all,
        avg_gower_valid=avg_gower_valid,
        avg_risk_before_pct=avg_risk_before_pct,
        avg_risk_after_pct=avg_risk_after_pct,
        min_risk_after_pct=min_risk_after_pct,
        risk_reduction_pct=risk_reduction_pct,
        total_gen_time_sec=total_gen_time_sec,
        avg_gen_time_sec=avg_gen_time_sec,
        top_features=top_features,
        sparsity_param=sparsity_param,
        locked_features=locked_features,
        csv_path=path_str,
    )


def _compute_avg_nchanged(cf_rows: pd.DataFrame) -> Optional[float]:
    """Compute average features changed per CF."""
    if len(cf_rows) == 0:
        return None
    if "Nchanged" in cf_rows.columns:
        values = pd.to_numeric(cf_rows["Nchanged"], errors="coerce")
        return round(values.mean(), 2) if not values.isna().all() else None
    # Fallback: count non-empty feature cells
    has_value = cf_rows[FEATURE_COLUMNS].notna()
    n_changed = has_value.sum(axis=1)
    return round(n_changed.mean(), 2)


def _compute_avg_gower(cf_rows: pd.DataFrame) -> Optional[float]:
    """Compute average Gower distance for CFs."""
    if len(cf_rows) == 0 or "gower_distance" not in cf_rows.columns:
        return None
    values = pd.to_numeric(cf_rows["gower_distance"], errors="coerce")
    mean_val = values.mean()
    return round(mean_val, 2) if not np.isnan(mean_val) else None


def _compute_avg_risk_before(original_rows: pd.DataFrame) -> Optional[float]:
    """Compute average baseline risk."""
    if len(original_rows) == 0 or "risk_before" not in original_rows.columns:
        return None
    mean_val = original_rows["risk_before"].mean()
    if np.isnan(mean_val):
        return None
    return round(mean_val * 100, 1)


def _compute_avg_risk_after(valid_cf_rows: pd.DataFrame) -> Optional[float]:
    """Compute average risk after CF (valid only)."""
    if len(valid_cf_rows) == 0 or "predicted_risk_after" not in valid_cf_rows.columns:
        return None
    mean_val = valid_cf_rows["predicted_risk_after"].mean()
    return round(mean_val * 100, 1) if not np.isnan(mean_val) else None


def _compute_min_risk_after(valid_cf_rows: pd.DataFrame) -> Optional[float]:
    """Compute minimum risk achieved (valid only)."""
    if len(valid_cf_rows) == 0 or "predicted_risk_after" not in valid_cf_rows.columns:
        return None
    min_val = valid_cf_rows["predicted_risk_after"].min()
    return round(min_val * 100, 1) if not np.isnan(min_val) else None


def _compute_risk_reduction(
    original_rows: pd.DataFrame, valid_cf_rows: pd.DataFrame
) -> Optional[float]:
    """Compute average risk reduction."""
    if len(valid_cf_rows) == 0:
        return None
    if (
        "risk_before" not in original_rows.columns
        or "predicted_risk_after" not in valid_cf_rows.columns
    ):
        return None

    reductions = []
    for qid in valid_cf_rows["query_index"].unique():
        original = original_rows[original_rows["query_index"] == qid]
        if len(original) == 0:
            continue
        risk_before = original["risk_before"].values[0]
        if np.isnan(risk_before) or risk_before == 0:
            continue

        person_valid = valid_cf_rows[valid_cf_rows["query_index"] == qid]
        for _, cf in person_valid.iterrows():
            risk_after = cf["predicted_risk_after"]
            if not np.isnan(risk_after):
                reduction = (risk_before - risk_after) / risk_before * 100
                reductions.append(reduction)

    if not reductions:
        return None
    return round(sum(reductions) / len(reductions), 1)


def _compute_actionable_pct(
    df: pd.DataFrame, original_rows: pd.DataFrame, cf_rows: pd.DataFrame
) -> Optional[float]:
    """
    Compute percentage of CFs that respect directional constraints (actionable).

    Actionable CFs are those that:
    - Increase features in SHOULD_INCREASE (cgtsmok, alcfreq, dosprt) OR keep them unchanged
    - Decrease features in SHOULD_DECREASE (bmi, etfruit, eatveg, slprl, paccnois) OR keep them unchanged

    This measures feasibility/actionability - whether the CF suggests realistic changes
    that align with improving health outcomes.
    """
    if len(cf_rows) == 0:
        return None

    actionable_count = 0
    total_count = 0

    for _, cf_row in cf_rows.iterrows():
        query_idx = cf_row["query_index"]

        # Get corresponding baseline (original) row
        baseline_rows = df[
            (df["query_index"] == query_idx) & (df["cf_id"] == "original")
        ]
        if len(baseline_rows) == 0:
            continue

        baseline = baseline_rows.iloc[0]
        is_actionable = True

        # Check features that should increase
        for feat in SHOULD_INCREASE:
            # Check if feature was changed (not empty string in CF)
            cf_val_str = cf_row.get(feat, "")
            if cf_val_str != "" and str(cf_val_str).strip() != "":
                # Feature was changed - check direction
                try:
                    baseline_val = float(baseline[feat])
                    cf_val = float(cf_val_str)

                    # Should increase but decreased - violation!
                    if cf_val < baseline_val:
                        is_actionable = False
                        break
                except (ValueError, TypeError):
                    # Skip if conversion fails
                    continue

        # Check features that should decrease
        if is_actionable:
            for feat in SHOULD_DECREASE:
                # Check if feature was changed (not empty string in CF)
                cf_val_str = cf_row.get(feat, "")
                if cf_val_str != "" and str(cf_val_str).strip() != "":
                    # Feature was changed - check direction
                    try:
                        baseline_val = float(baseline[feat])
                        cf_val = float(cf_val_str)

                        # Should decrease but increased - violation!
                        if cf_val > baseline_val:
                            is_actionable = False
                            break
                    except (ValueError, TypeError):
                        # Skip if conversion fails
                        continue

        if is_actionable:
            actionable_count += 1
        total_count += 1

    if total_count == 0:
        return None

    return round(actionable_count / total_count * 100, 1)


def _compute_top_features(valid_cf_rows: pd.DataFrame) -> str:
    """Compute top 4 most changed features."""
    if len(valid_cf_rows) == 0:
        return ""

    n_valid = len(valid_cf_rows)
    frequencies = []

    for feature in FEATURE_COLUMNS:
        n_changed = valid_cf_rows[feature].notna().sum()
        pct = n_changed / n_valid * 100
        if pct > 0:
            frequencies.append((feature, pct))

    # Sort and return top 4
    sorted_features = sorted(frequencies, key=lambda x: x[1], reverse=True)
    top_4 = sorted_features[:4]

    if not top_4:
        return ""

    return "  |  ".join(f"{feat}: {pct:.0f}%" for feat, pct in top_4)


def generate_comparison_report(summaries: list[ExperimentSummary]) -> pd.DataFrame:
    """
    Convert experiment summaries to DataFrame with proper column ordering.

    Args:
        summaries: List of ExperimentSummary objects

    Returns:
        DataFrame with one row per experiment
    """
    rows = []
    for summary in summaries:
        row = {
            "experiment": summary.experiment,
            "explainer_type": summary.explainer_type,
            "ml_model_type": summary.ml_model_type,
            "use_permitted_range": _format_bool(summary.use_permitted_range),
            "total_cfs_requested": summary.total_cfs_requested,
            "maxiterations": summary.maxiterations,
            "stopping_threshold": f"{summary.stopping_threshold:.2f}",
            "n_patients": summary.n_patients,
            "total_cfs": summary.total_cfs,
            "valid_cfs": summary.valid_cfs,
            "validity_%": f"{summary.validity_pct:.1f}%",
            "solved_%": f"{summary.solved_pct:.1f}%",
            "actionable_%": _format_float(summary.actionable_pct, 1) + "%"
            if summary.actionable_pct is not None
            else "",
            "avg_nchanged": _format_float(summary.avg_nchanged, 2),
            "avg_nchanged_all": _format_float(summary.avg_nchanged_all, 2),
            "avg_gower_valid": _format_float(summary.avg_gower_valid, 2),
            "avg_risk_before_%": _format_float(summary.avg_risk_before_pct, 1),
            "avg_risk_after_%": _format_float(summary.avg_risk_after_pct, 1),
            "min_risk_after_%": _format_float(summary.min_risk_after_pct, 1),
            "risk_reduction_%": _format_float(summary.risk_reduction_pct, 1),
            "total_gen_time_sec": f"{summary.total_gen_time_sec:.2f}",
            "avg_gen_time_sec": f"{summary.avg_gen_time_sec:.2f}",
            "top_features": summary.top_features,
        }

        # Optional columns
        if summary.sparsity_param is not None:
            row["sparsity_param"] = f"{summary.sparsity_param:.2f}"
        if summary.locked_features is not None:
            row["locked_features"] = summary.locked_features

        row["csv_path"] = summary.csv_path
        rows.append(row)

    return pd.DataFrame(rows)


def _format_bool(value: Optional[bool]) -> str:
    """Format boolean value for CSV."""
    if value is None:
        return ""
    return "True" if value else "False"


def _format_float(value: Optional[float], decimals: int) -> str:
    """Format float value for CSV."""
    if value is None:
        return ""
    return f"{value:.{decimals}f}"
