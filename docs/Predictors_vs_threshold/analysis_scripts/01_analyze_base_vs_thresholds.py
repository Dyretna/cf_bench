"""
Analyze Base Models vs Thresholds Results
==========================================
Analyzes CF generation results for baseline RF and XGB models
with different stopping thresholds.

Author: Generated for Generation 2 Analysis
Date: May 7, 2026
"""

from pathlib import Path

import pandas as pd


def analyze_results(folder_path, model_name, threshold_name):
    """
    Analyze a single experiment folder.

    Parameters
    ----------
    folder_path : Path
        Path to experiment output folder
    model_name : str
        Model name (e.g., 'RF', 'XGB')
    threshold_name : str
        Threshold name (e.g., 'highthres', 'midthres', 'lowthres')

    Returns
    -------
    dict
        Dictionary with analysis metrics
    """
    csv_path = folder_path / "annotated.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # Get only CF rows (not original)
    cf_rows = df[df["cf_id"] != "original"]

    # Overall metrics
    total_cfs = len(cf_rows)
    valid_cfs = cf_rows["valid"].sum()

    # Per-query metrics
    queries = df[df["cf_id"] == "original"]["query_index"].unique()
    n_queries = len(queries)

    # Success rate per query (at least 1 valid CF)
    query_success = []
    for q in queries:
        query_cfs = cf_rows[cf_rows["query_index"] == q]
        has_valid = query_cfs["valid"].sum() > 0
        query_success.append(has_valid)

    query_success_rate = sum(query_success) / len(query_success) if query_success else 0

    # Average metrics for valid CFs
    valid_df = cf_rows[cf_rows["valid"]]

    if len(valid_df) > 0:
        avg_gower = valid_df["gower_distance"].mean()
        avg_nchanged = valid_df["Nchanged"].mean()
        avg_gen_time = df[df["cf_id"] == "original"]["cf_gen_time_sec"].mean()
    else:
        avg_gower = None
        avg_nchanged = None
        avg_gen_time = df[df["cf_id"] == "original"]["cf_gen_time_sec"].mean()

    # Read config to get threshold value
    config_files = list(folder_path.glob("config_*.txt"))
    stopping_threshold = None
    if config_files:
        with open(config_files[0], "r") as f:
            for line in f:
                if "stopping_threshold" in line:
                    stopping_threshold = line.split(":")[1].strip()
                    break

    return {
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
    # Path to results
    base_path = Path("cf_outputs/base_vs_thresholds")

    if not base_path.exists():
        print(f"ERROR: Path {base_path} does not exist!")
        print("Make sure you're running this from the project root directory.")
        return

    results = []

    # Analyze each folder
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir():
            parts = folder.name.split("_")
            model = parts[1]  # rf or xgb
            threshold = parts[2]  # highthres, midthres, lowthres

            result = analyze_results(folder, model.upper(), threshold)
            if result:
                results.append(result)

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("BASE VS THRESHOLDS RESULTS")
    print("=" * 80 + "\n")
    print(df_results.to_string(index=False))

    # Save to CSV
    output_path = Path(
        "docs/Predictors_vs_threshold/analysis_scripts/output/base_vs_thresholds_summary.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nBy Model:")
    print(
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

    print("\nBy Threshold:")
    print(
        df_results.groupby("threshold")
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


if __name__ == "__main__":
    main()
