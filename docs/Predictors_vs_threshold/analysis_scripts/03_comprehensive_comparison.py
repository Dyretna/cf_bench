"""
Comprehensive Comparison of All Experiments
============================================
Combines all experiment results and generates rankings and recommendations.

Author: Generated for Generation 2 Analysis
Date: May 7, 2026
"""

from pathlib import Path

import pandas as pd


def load_summaries():
    """Load previously generated summary CSVs."""
    base_df = pd.read_csv(
        "docs/Predictors_vs_threshold/analysis_scripts/output/base_vs_thresholds_summary.csv"
    )
    smote_xgb_df = pd.read_csv(
        "docs/Predictors_vs_threshold/analysis_scripts/output/smote_xgb_optimized_summary.csv"
    )
    return base_df, smote_xgb_df


def print_table_section(title, df, columns):
    """Print a formatted table section."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(df[columns].to_string(index=False))


def main():
    """Main comparison function."""
    # Load data
    base_df, smote_xgb_df = load_summaries()

    print("=" * 90)
    print("COMPREHENSIVE COMPARISON OF ALL EXPERIMENTS")
    print("=" * 90)

    # Section 1: Base Models
    print("\n\n1. BASE MODELS (RF & XGB) - Different Thresholds")
    print("-" * 90)

    for _, row in base_df.iterrows():
        print(
            f"{row['model']:<8} {row['threshold']:<12} {row['n_queries']:<8} "
            f"{row['valid_cfs']:<10.0f} {row['validity_rate'] * 100:<12.1f} "
            f"{row['query_success_rate'] * 100:<10.1f} {row['avg_gower_distance']:<8.3f} "
            f"{row['avg_features_changed']:<10.2f} {row['avg_gen_time_sec']:<8.2f}"
        )

    # Section 2: SMOTE & XGB Optimized
    print("\n\n2. SMOTE & XGB OPTIMIZED - Different Configurations")
    print("-" * 90)

    for _, row in smote_xgb_df.iterrows():
        print(
            f"{row['experiment']:<16} {row['model']:<14} {row['threshold']:<12} "
            f"{row['valid_cfs']:<10.0f} {row['validity_rate'] * 100:<12.1f} "
            f"{row['query_success_rate'] * 100:<10.1f} {row['avg_gower_distance']:<8.3f} "
            f"{row['avg_features_changed']:<10.2f} {row['avg_gen_time_sec']:<8.2f}"
        )

    # Combine all results
    all_results = []

    for _, row in base_df.iterrows():
        all_results.append(
            {
                "config": f"Base {row['model']} - {row['threshold']}",
                "experiment": "Baseline",
                "validity_rate": row["validity_rate"],
                "success_rate": row["query_success_rate"],
                "valid_cfs": row["valid_cfs"],
                "gower": row["avg_gower_distance"],
                "changes": row["avg_features_changed"],
                "time": row["avg_gen_time_sec"],
            }
        )

    for _, row in smote_xgb_df.iterrows():
        all_results.append(
            {
                "config": f"{row['experiment']} {row['model']} - {row['threshold']}",
                "experiment": row["experiment"],
                "validity_rate": row["validity_rate"],
                "success_rate": row["query_success_rate"],
                "valid_cfs": row["valid_cfs"],
                "gower": row["avg_gower_distance"],
                "changes": row["avg_features_changed"],
                "time": row["avg_gen_time_sec"],
            }
        )

    df_all = pd.DataFrame(all_results)

    print("\n\n" + "=" * 90)
    print("KEY FINDINGS - RANKED BY PERFORMANCE")
    print("=" * 90)

    # Rankings
    print("\n\nTop 5 by Query Success Rate (at least 1 valid CF per query):")
    print("-" * 90)
    top_success = df_all.sort_values("success_rate", ascending=False).head(5)
    for i, (_, row) in enumerate(top_success.iterrows(), 1):
        print(
            f"{i}. {row['config']:<50} Success: {row['success_rate'] * 100:>5.1f}%  "
            f"Valid CFs: {row['valid_cfs']:.0f}"
        )

    print("\n\nTop 5 by Validity Rate (proportion of valid CFs among generated):")
    print("-" * 90)
    top_validity = df_all.sort_values("validity_rate", ascending=False).head(5)
    for i, (_, row) in enumerate(top_validity.iterrows(), 1):
        print(
            f"{i}. {row['config']:<50} Validity: {row['validity_rate'] * 100:>5.1f}%  "
            f"Valid CFs: {row['valid_cfs']:.0f}"
        )

    print("\n\nTop 5 by Number of Valid CFs Generated:")
    print("-" * 90)
    top_valid = df_all.sort_values("valid_cfs", ascending=False).head(5)
    for i, (_, row) in enumerate(top_valid.iterrows(), 1):
        print(
            f"{i}. {row['config']:<50} Valid CFs: {row['valid_cfs']:>5.0f}  "
            f"Validity: {row['validity_rate'] * 100:>5.1f}%"
        )

    print("\n\nFastest Generation (Average time per query):")
    print("-" * 90)
    fastest = df_all.sort_values("time", ascending=True).head(5)
    for i, (_, row) in enumerate(fastest.iterrows(), 1):
        print(
            f"{i}. {row['config']:<50} Time: {row['time']:>6.2f}s  "
            f"Success: {row['success_rate'] * 100:>5.1f}%"
        )

    # Overall recommendation
    print("\n\n" + "=" * 90)
    print("OVERALL RECOMMENDATION")
    print("=" * 90)

    df_baseline_only = df_all[df_all["experiment"] == "Baseline"]
    best_baseline = df_baseline_only.sort_values(
        ["success_rate", "validity_rate", "valid_cfs"], ascending=[False, False, False]
    ).iloc[0]

    print("\nBest Baseline Model:")
    print(f"  Configuration: {best_baseline['config']}")
    print(f"  Query Success Rate: {best_baseline['success_rate'] * 100:.1f}%")
    print(f"  Validity Rate: {best_baseline['validity_rate'] * 100:.1f}%")
    print(f"  Valid CFs Generated: {best_baseline['valid_cfs']:.0f}")
    print(f"  Avg Generation Time: {best_baseline['time']:.2f}s")
    print("\n  Why: Consistent 100% query success, good validity rate, fast generation")

    df_xgb_opt = df_all[df_all["experiment"] == "XGB-optimized"]
    if len(df_xgb_opt) > 0:
        best_xgb_opt = df_xgb_opt.sort_values(
            ["success_rate", "validity_rate", "valid_cfs"],
            ascending=[False, False, False],
        ).iloc[0]
        print("\nBest XGB-Optimized:")
        print(f"  Configuration: {best_xgb_opt['config']}")
        print(f"  Query Success Rate: {best_xgb_opt['success_rate'] * 100:.1f}%")
        print(f"  Validity Rate: {best_xgb_opt['validity_rate'] * 100:.1f}%")
        print(f"  Valid CFs Generated: {best_xgb_opt['valid_cfs']:.0f}")
        print(f"  Avg Generation Time: {best_xgb_opt['time']:.2f}s")
        print("\n  Why: Hyperparameter-optimized structure, high success rate")

    # Save combined results
    output_path = Path(
        "docs/Predictors_vs_threshold/analysis_scripts/output/all_experiments_combined.csv"
    )
    df_all.to_csv(output_path, index=False)
    print(f"\n\nCombined results saved to: {output_path}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
