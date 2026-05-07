"""
Analyze SMOTE and XGB Optimized Results
========================================
Analyzes CF generation results for SMOTE-trained models and
XGB models with optimized hyperparameters.

Author: Generated for Generation 2 Analysis
Date: May 7, 2026
"""

from pathlib import Path

import pandas as pd


def analyze_results(folder_path, model_name, threshold_name, experiment_type):
    """
    Analyze a single experiment folder.

    Parameters
    ----------
    folder_path : Path
        Path to experiment output folder
    model_name : str
        Model name (e.g., 'RandomForest', 'XGBoost')
    threshold_name : str
        Threshold name (e.g., 'thres0.5', 'thres0.9')
    experiment_type : str
        Experiment type (e.g., 'SMOTE-base', 'SMOTE-grid', 'XGB-optimized')

    Returns
    -------
    dict
        Dictionary with analysis metrics
    """
    csv_path = folder_path / "annotated.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    cf_rows = df[df["cf_id"] != "original"]

    total_cfs = len(cf_rows)
    valid_cfs = cf_rows["valid"].sum()

    queries = df[df["cf_id"] == "original"]["query_index"].unique()
    n_queries = len(queries)

    query_success = []
    for q in queries:
        query_cfs = cf_rows[cf_rows["query_index"] == q]
        has_valid = query_cfs["valid"].sum() > 0
        query_success.append(has_valid)

    query_success_rate = sum(query_success) / len(query_success) if query_success else 0

    valid_df = cf_rows[cf_rows["valid"]]

    if len(valid_df) > 0:
        avg_gower = valid_df["gower_distance"].mean()
        avg_nchanged = valid_df["Nchanged"].mean()
        avg_gen_time = df[df["cf_id"] == "original"]["cf_gen_time_sec"].mean()
    else:
        avg_gower = None
        avg_nchanged = None
        avg_gen_time = df[df["cf_id"] == "original"]["cf_gen_time_sec"].mean()

    config_files = list(folder_path.glob("config_*.txt"))
    stopping_threshold = None
    if config_files:
        with open(config_files[0], "r") as f:
            for line in f:
                if "stopping_threshold" in line:
                    stopping_threshold = line.split(":")[1].strip()
                    break

    return {
        "experiment": experiment_type,
        "model": model_name,
        "threshold": threshold_name,
        "stopping_threshold": stopping_threshold,
        "n_queries": n_queries,
        "total_cfs_generated": total_cfs,
        "valid_cfs": valid_cfs,
        "validity_rate": valid_cfs / total_cfs if total_cfs > 0 else 0,
        "query_success_rate": query_success_rate,
        "avg_gower_distance": avg_gower,
        "avg_features_changed": avg_nchanged,
        "avg_gen_time_sec": avg_gen_time,
    }


def main():
    """Main analysis function."""
    # Paths
    smote_path = Path("cf_outputs/SMOTE")
    xgb_opt_path = Path("cf_outputs/xgb_optimized")

    results = []

    # Analyze SMOTE base predictors
    base_pred_path = smote_path / "base_predictors"
    if base_pred_path.exists():
        for folder in sorted(base_pred_path.iterdir()):
            if folder.is_dir():
                parts = folder.name.split("_")
                model = parts[0]
                threshold = parts[1]
                result = analyze_results(folder, model, threshold, "SMOTE-base")
                if result:
                    results.append(result)

    # Analyze SMOTE gridsearched predictors
    grid_pred_path = smote_path / "gridsearched_predictors"
    if grid_pred_path.exists():
        for folder in sorted(grid_pred_path.iterdir()):
            if folder.is_dir():
                parts = folder.name.split("_")
                model = parts[0]
                threshold = parts[1]
                result = analyze_results(folder, model, threshold, "SMOTE-grid")
                if result:
                    results.append(result)

    # Analyze XGB optimized
    if xgb_opt_path.exists():
        for folder in sorted(xgb_opt_path.iterdir()):
            if folder.is_dir():
                parts = folder.name.split("_")
                model = parts[0]
                threshold = parts[1]
                result = analyze_results(folder, model, threshold, "XGB-optimized")
                if result:
                    results.append(result)

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("SMOTE AND XGB OPTIMIZED RESULTS")
    print("=" * 80 + "\n")
    print(df_results.to_string(index=False))

    # Save to CSV
    output_path = Path(
        "docs/Predictors_vs_threshold/analysis_scripts/output/smote_xgb_optimized_summary.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nBy Experiment Type:")
    summary = (
        df_results.groupby("experiment")
        .agg(
            {
                "query_success_rate": ["mean", "min", "max"],
                "validity_rate": ["mean", "min", "max"],
                "valid_cfs": "sum",
                "avg_gen_time_sec": "mean",
            }
        )
        .round(3)
    )
    print(summary)

    print("\nBy Model (across all experiments):")
    summary_model = (
        df_results.groupby("model")
        .agg(
            {
                "query_success_rate": "mean",
                "validity_rate": "mean",
                "valid_cfs": "sum",
                "avg_gen_time_sec": "mean",
            }
        )
        .round(3)
    )
    print(summary_model)


if __name__ == "__main__":
    main()
