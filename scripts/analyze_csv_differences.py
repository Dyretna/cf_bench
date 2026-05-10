#!/usr/bin/env python3
"""
Analyze annotated.csv files to determine migration needs.

This script scans all annotated.csv files and reports:
- Which columns exist
- Column ordering
- Whether Nchanged is present and populated
- Whether gower_distance is present and populated
- Which files need which migrations

Usage:
    python scripts/analyze_csv_differences.py --all
    python scripts/analyze_csv_differences.py --dir gen_1_experiments
    python scripts/analyze_csv_differences.py --all --report migration_report.txt
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

EXPECTED_COLUMNS = [
    "query_index",
    "cf_id",
    "etfruit",
    "eatveg",
    "cgtsmok",
    "alcfreq",
    "slprl",
    "paccnois",
    "bmi",
    "dosprt",
    "cf_gen_time_sec",
    "gower_distance",
    "Nchanged",
    "valid",
    "risk_before",
    "predicted_risk_after",
    "target_risk",
]


def analyze_single_file(csv_path: Path) -> dict:
    """
    Analyze a single annotated.csv file.

    Returns:
        Dictionary with analysis results
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "columns": [],
            "needs_migration": True,
        }

    columns = df.columns.tolist()
    cf_rows = df[df["cf_id"] != "original"] if "cf_id" in df.columns else df

    # Check Nchanged
    has_nchanged = "Nchanged" in columns
    nchanged_populated = False
    if has_nchanged and len(cf_rows) > 0:
        nchanged_populated = cf_rows["Nchanged"].notna().any()

    # Check gower_distance
    has_gower = "gower_distance" in columns
    gower_populated = False
    if has_gower and len(cf_rows) > 0:
        gower_vals = cf_rows["gower_distance"]
        gower_populated = gower_vals.notna().any() and not (gower_vals == "").all()

    # Check column order
    expected_order_subset = [col for col in EXPECTED_COLUMNS if col in columns]
    actual_order_subset = [col for col in columns if col in EXPECTED_COLUMNS]
    correct_order = expected_order_subset == actual_order_subset

    # Determine what needs fixing
    needs = []
    if not has_nchanged:
        needs.append("add_nchanged")
    elif not nchanged_populated:
        needs.append("populate_nchanged")

    if not has_gower:
        needs.append("add_gower")
    elif not gower_populated:
        needs.append("populate_gower")

    if not correct_order:
        needs.append("reorder_columns")

    return {
        "status": "ok",
        "columns": columns,
        "column_count": len(columns),
        "has_nchanged": has_nchanged,
        "nchanged_populated": nchanged_populated,
        "has_gower": has_gower,
        "gower_populated": gower_populated,
        "correct_order": correct_order,
        "needs_migration": len(needs) > 0,
        "migration_needs": needs,
        "row_count": len(df),
        "cf_row_count": len(cf_rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze annotated.csv files for migration needs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all
  %(prog)s --dir gen_1_experiments
  %(prog)s --all --report migration_report.txt
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="Analyze all annotated.csv in cf_outputs"
    )
    parser.add_argument(
        "--dir", type=str, help="Analyze all annotated.csv in a specific directory"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="cf_outputs",
        help="Base path for cf_outputs directory",
    )
    parser.add_argument("--report", type=str, help="Save detailed report to file")

    args = parser.parse_args()

    if not (args.all or args.dir):
        parser.error("Please specify --all or --dir")

    # Resolve base path
    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist")
        return 1

    # Find files
    if args.all:
        files = list(base_path.glob("**/annotated.csv"))
    else:
        dir_path = base_path / args.dir
        if not dir_path.exists():
            print(f"Error: Directory '{dir_path}' does not exist")
            return 1
        files = list(dir_path.glob("**/annotated.csv"))

    # Exclude backup and failed files
    files = [f for f in files if ".backup" not in f.name and ".failed" not in f.name]

    if not files:
        print("No annotated.csv files found!")
        return 1

    print(f"\nAnalyzing {len(files)} annotated.csv file(s)...\n")

    # Analyze each file
    results = {}
    for csv_path in sorted(files):
        try:
            rel_path = csv_path.relative_to(base_path)
        except ValueError:
            rel_path = csv_path
        results[str(rel_path)] = analyze_single_file(csv_path)

    # Aggregate statistics
    total_files = len(results)
    error_files = sum(1 for r in results.values() if r["status"] == "error")
    needs_migration = sum(
        1 for r in results.values() if r.get("needs_migration", False)
    )
    ready_files = total_files - needs_migration - error_files

    # Group by migration needs
    migration_groups = defaultdict(list)
    for path, result in results.items():
        if result["status"] == "error":
            migration_groups["error"].append(path)
        elif not result.get("needs_migration", False):
            migration_groups["ready"].append(path)
        else:
            needs_key = tuple(sorted(result.get("migration_needs", [])))
            migration_groups[needs_key].append(path)

    # Print summary
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total files analyzed: {total_files}")
    print(f"Files ready (no migration needed): {ready_files}")
    print(f"Files needing migration: {needs_migration}")
    print(f"Files with errors: {error_files}")
    print()

    print("Migration Groups:")
    print("-" * 70)

    # Sort groups - handle mixed types (tuples and strings)
    sorted_groups = []
    for needs, paths in migration_groups.items():
        if needs == "error":
            sort_key = (0, "error")
        elif needs == "ready":
            sort_key = (1, "ready")
        else:
            sort_key = (2, str(needs))
        sorted_groups.append((sort_key, needs, paths))

    for _, needs, paths in sorted(sorted_groups):
        if needs == "error":
            print(f"\n[ERROR] {len(paths)} file(s):")
        elif needs == "ready":
            print(f"\n[READY] {len(paths)} file(s) - no migration needed")
        else:
            needs_str = ", ".join(needs)
            print(f"\n[{needs_str.upper()}] {len(paths)} file(s):")

        for path in sorted(paths)[:5]:  # Show first 5
            print(f"  - {path}")
        if len(paths) > 5:
            print(f"  ... and {len(paths) - 5} more")

    # Detailed report if requested
    if args.report:
        report_path = Path(args.report)
        with open(report_path, "w") as f:
            f.write("ANNOTATED CSV MIGRATION ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total files: {total_files}\n")
            f.write(f"Ready: {ready_files}\n")
            f.write(f"Need migration: {needs_migration}\n")
            f.write(f"Errors: {error_files}\n\n")

            for _, needs, paths in sorted(sorted_groups):
                if needs == "error":
                    f.write(f"\nERROR GROUP ({len(paths)} files):\n")
                elif needs == "ready":
                    f.write(f"\nREADY GROUP ({len(paths)} files):\n")
                else:
                    needs_str = ", ".join(needs)
                    f.write(f"\n{needs_str.upper()} GROUP ({len(paths)} files):\n")

                f.write("-" * 70 + "\n")
                for path in sorted(paths):
                    f.write(f"{path}\n")
                    result = results[path]
                    if result["status"] == "ok":
                        f.write(f"  Columns: {result['column_count']}\n")
                        f.write(
                            f"  Rows: {result['row_count']} ({result['cf_row_count']} CF rows)\n"
                        )
                        f.write(
                            f"  Nchanged: {'Yes' if result['has_nchanged'] else 'No'}"
                        )
                        if result["has_nchanged"]:
                            f.write(
                                f" ({'populated' if result['nchanged_populated'] else 'empty'})"
                            )
                        f.write("\n")
                        f.write(f"  Gower: {'Yes' if result['has_gower'] else 'No'}")
                        if result["has_gower"]:
                            f.write(
                                f" ({'populated' if result['gower_populated'] else 'empty'})"
                            )
                        f.write("\n")
                        f.write(
                            f"  Column order: {'Correct' if result['correct_order'] else 'Needs reordering'}\n"
                        )
                    else:
                        f.write(f"  Error: {result.get('error', 'Unknown')}\n")
                f.write("\n")

        print(f"\nDetailed report saved to: {report_path}")

    print("\n" + "=" * 70)

    if needs_migration > 0:
        print(f"\nNext step: Run migration script on {needs_migration} file(s)")
        print("  python scripts/migrate_annotated_results.py --dir <path>")
    else:
        print("\nAll files are ready - no migration needed!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
