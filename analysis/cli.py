"""
Simplified CLI for counterfactual experiment analysis.

Following the architecture from Results_Summary_Analysis_Plan.md:
- Functional approach with dataclasses
- Auto-discovery of experiments
- Config extraction from files
- Clean, simple code

Usage:
    python cli.py --all
    python cli.py --dir predictors_vs_threshold/baseline
    python cli.py --pattern "**/XGBoost**/annotated.csv"
    python cli.py --all --output results.csv
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .reporter.summary import generate_comparison_report, summarize_experiment

# Load environment variables from .env
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze counterfactual experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all
  %(prog)s --dir predictors_vs_threshold/baseline --output results.csv
  %(prog)s --pattern "**/XGBoost**/annotated.csv"
        """,
    )

    # Input options
    parser.add_argument(
        "--base-path",
        type=str,
        default=os.getenv("CF_OUTPUTS", "../cf_outputs"),
        help="Base path for cf_outputs directory (default: CF_OUTPUTS env var or ../cf_outputs)",
    )
    parser.add_argument(
        "--dir", type=str, help="Analyze all experiments in a specific directory"
    )
    parser.add_argument(
        "--pattern", type=str, help="Glob pattern to find annotated.csv files"
    )
    parser.add_argument(
        "--all", action="store_true", help="Analyze all experiments in cf_outputs"
    )

    # Output options
    parser.add_argument("--output", type=str, help="Save results to CSV file")
    parser.add_argument(
        "--include-config",
        action="store_true",
        help="Include sparsity and locking columns",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Don't print report to console"
    )

    args = parser.parse_args()

    # Validate arguments
    if not (args.pattern or args.dir or args.all):
        parser.error("Please specify --pattern, --dir, or --all")

    # Initialize base path
    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist")
        return 1

    # Build search pattern
    if args.all:
        pattern = "**/annotated.csv"
        print("Loading all experiments...")
    elif args.dir:
        pattern = f"{args.dir}/**/annotated.csv"
        print(f"Loading experiments from {args.dir}")
    else:
        pattern = args.pattern
        print(f"Loading experiments matching pattern: {pattern}")

    # Find all experiment CSV files
    csv_files = list(base_path.glob(pattern))

    if not csv_files:
        print("No experiments found!")
        return 1

    print(f"✓ Found {len(csv_files)} experiment(s)\n")

    # Summarize each experiment
    summaries = []
    for csv_path in sorted(csv_files):
        try:
            summary = summarize_experiment(
                csv_path, base_path=base_path, include_constraints=args.include_config
            )
            summaries.append(summary)
        except Exception as e:
            print(f"Warning: Failed to process {csv_path}: {e}")

    if not summaries:
        print("No experiments could be processed!")
        return 1

    # Generate comparison report
    df = generate_comparison_report(summaries)

    # Print to console (unless --quiet)
    if not args.quiet:
        print("\n" + "=" * 100)
        print("COUNTERFACTUAL EXPERIMENT RESULTS — COMPARISON TABLE")
        print("=" * 100 + "\n")

        for _, row in df.iterrows():
            print(f"--- {row['experiment']} ---")
            for col in df.columns:
                if col != "experiment":
                    print(f"  {col:30s}: {row[col]}")
            print()

        print("=" * 100 + "\n")

    # Save to CSV (if requested)
    if args.output:
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    main()
