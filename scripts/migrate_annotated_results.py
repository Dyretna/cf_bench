#!/usr/bin/env python3
"""
Migrate annotated.csv files to current standardized format.

This is the UNIVERSAL migration tool for all annotated.csv files.
It intelligently detects what operations are needed and safely migrates by:
1. Adding and computing Nchanged column (if missing or empty)
2. Computing/recomputing gower_distance values (handles NaN vs empty string)
3. Reordering columns to match current standard format

Safety features:
- Creates .migration_backup before modifying files
- Preserves original dtypes (ordinal features stay as float64 due to NaN)
- Skips files that already have correct format
- Dry-run mode to preview changes
- Handles both old format (NaN for unchanged) and new format (empty strings)

Usage:
    python scripts/migrate_annotated_results.py --all
    python scripts/migrate_annotated_results.py --dir gen_1_experiments
    python scripts/migrate_annotated_results.py --dir predictors_vs_threshold
    python scripts/migrate_annotated_results.py --file cf_outputs/path/annotated.csv
    python scripts/migrate_annotated_results.py --all --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cf_bench.results.metrics import PerformanceMetrics

# Load environment variables
load_dotenv()

# Feature types from SystemConfig (config/config.py)
# Feature order MUST match config.py feature_cols exactly
FEATURE_COLUMNS = [
    "etfruit",
    "eatveg",
    "cgtsmok",
    "alcfreq",
    "slprl",
    "paccnois",
    "bmi",  # Continuous feature (full float precision)
    "dosprt",
]

# ORDINAL features should be written as integers without .0 suffix
ORDINAL_FEATURES = [
    "etfruit",
    "eatveg",
    "cgtsmok",
    "alcfreq",
    "slprl",
    "paccnois",
    "dosprt",
]

# CONTINUOUS features keep full float precision
CONTINUOUS_FEATURES = ["bmi"]


def compute_nchanged(row, feature_cols):
    """Count number of changed features in a CF row."""
    if row.get("cf_id") == "original":
        return None

    count = 0
    for col in feature_cols:
        val = row.get(col)
        if pd.notna(val) and val != "" and str(val).strip() != "":
            count += 1
    return float(count) if count > 0 else 0.0


def migrate_single_file(csv_path: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single annotated.csv file.

    Args:
        csv_path: Path to annotated.csv
        dry_run: Only report what would be done

    Returns:
        True if migrated successfully, False otherwise
    """
    if not csv_path.exists():
        print(f"  [ERROR] File not found: {csv_path}")
        return False

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}")
        return False

    changes_made = []

    # Check what needs to be done
    needs_nchanged = "Nchanged" not in df.columns
    has_gower = "gower_distance" in df.columns
    cf_rows = df[df["cf_id"] != "original"] if "cf_id" in df.columns else df

    if has_gower and len(cf_rows) > 0:
        gower_vals = cf_rows["gower_distance"]
        needs_gower = gower_vals.isna().all() or (gower_vals == "").all()
    else:
        needs_gower = not has_gower

    if dry_run:
        if needs_nchanged:
            print("  [DRY-RUN] Would add Nchanged column")
        if needs_gower:
            action = "compute" if has_gower else "add and compute"
            print(f"  [DRY-RUN] Would {action} gower_distance")
        print("  [DRY-RUN] Would reorder columns")
        return True

    # Create backup if not exists
    backup_path = csv_path.parent / f"{csv_path.name}.migration_backup"
    if not backup_path.exists():
        try:
            shutil.copy2(csv_path, backup_path)
            print(f"  [OK] Backup created: {backup_path.name}")
        except Exception as e:
            print(f"  [ERROR] Failed to create backup: {e}")
            return False

    # 1. Add/compute Nchanged
    if needs_nchanged:
        try:
            df["Nchanged"] = df.apply(
                lambda row: compute_nchanged(row, FEATURE_COLUMNS), axis=1
            )
            changes_made.append("added_nchanged")
            print("  [OK] Added and computed Nchanged column")
        except Exception as e:
            print(f"  [ERROR] Failed to compute Nchanged: {e}")
            return False

    # 2. Compute gower_distance
    if needs_gower:
        try:
            # Drop existing empty column if present
            if has_gower:
                df = df.drop(columns=["gower_distance"])

            # Validate required columns exist
            required_cols = ["query_index", "cf_id"] + FEATURE_COLUMNS
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  [ERROR] Missing columns for gower: {missing_cols}")
                return False

            # Convert NaN to empty strings in feature columns for CF rows
            # Old files use NaN for unchanged features, new files use ""
            cf_mask = df["cf_id"] != "original"
            for col in FEATURE_COLUMNS:
                if col in df.columns:
                    df.loc[cf_mask, col] = df.loc[cf_mask, col].fillna("")

            df = PerformanceMetrics.add_gower_distance_column(
                annotated_batch=df,
                feature_cols=FEATURE_COLUMNS,
                query_index_col="query_index",
                cf_id_col="cf_id",
            )
            changes_made.append("computed_gower")
            print("  [OK] Computed gower_distance values")
        except Exception as e:
            print(f"  [ERROR] Failed to compute gower_distance: {e}")
            return False

    # 3. Reorder columns to standard format
    try:
        ordered_cols = []

        # Metadata columns
        if "query_index" in df.columns:
            ordered_cols.append("query_index")
        if "cf_id" in df.columns:
            ordered_cols.append("cf_id")

        # Feature columns in standard order
        for feat in FEATURE_COLUMNS:
            if feat in df.columns:
                ordered_cols.append(feat)

        # Metric columns in standard order
        metric_cols = [
            "cf_gen_time_sec",
            "gower_distance",
            "Nchanged",
            "valid",
            "risk_before",
            "predicted_risk_after",
            "target_risk",
        ]
        for col in metric_cols:
            if col in df.columns and col not in ordered_cols:
                ordered_cols.append(col)

        # Add any remaining columns (e.g., hltprhc target)
        for col in df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)

        df = df[ordered_cols]
        changes_made.append("reordered_columns")
        print("  [OK] Reordered columns to standard format")
    except Exception as e:
        print(f"  [ERROR] Failed to reorder columns: {e}")
        return False

    # 4. Save migrated file matching original CSV format
    try:
        # Apply rounding to match original pipeline formatting:
        # - gower_distance: 4 decimals
        # - cf_gen_time_sec: 2 decimals
        # - risk columns: 4 decimals
        # - bmi: 5 decimals
        # - Nchanged: 1 decimal (1.0, 2.0, etc.)

        # Round gower_distance to 4 decimals
        if "gower_distance" in df.columns:
            df["gower_distance"] = df["gower_distance"].apply(
                lambda x: round(float(x), 4) if pd.notna(x) and x != "" else x
            )

        # Round cf_gen_time_sec to 2 decimals
        if "cf_gen_time_sec" in df.columns:
            df["cf_gen_time_sec"] = df["cf_gen_time_sec"].apply(
                lambda x: round(float(x), 2) if pd.notna(x) and x != "" else x
            )

        # Round risk columns to 4 decimals
        for col in ["risk_before", "predicted_risk_after", "target_risk"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: round(float(x), 4) if pd.notna(x) and x != "" else x
                )

        # Round BMI to 5 decimals
        if "bmi" in df.columns:
            df["bmi"] = df["bmi"].apply(
                lambda x: round(float(x), 5) if pd.notna(x) and x != "" else x
            )

        # Format features according to SystemConfig types:
        # - ORDINAL features: integers without .0 (e.g., 4, 3, 7)
        # - CONTINUOUS features: keep full precision (e.g., 18.9865907203038)

        # Convert ordinal features to integers (remove .0)
        for col in ["query_index"] + ORDINAL_FEATURES:
            if col in df.columns:
                # Replace NaN with empty string
                df[col] = df[col].fillna("")
                # Convert to string, removing .0 from integers
                df[col] = df[col].apply(
                    lambda x: str(int(float(x)))
                    if x != ""
                    and pd.notna(x)
                    and str(x) != "nan"
                    and float(x) == int(float(x))
                    else (x if x == "" else str(x))
                )

        # Continuous features (BMI) keep float precision, just fill NaN with empty string
        for col in CONTINUOUS_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna("")
        # valid is already object/boolean

        df.to_csv(csv_path, index=False)
        changes_str = ", ".join(changes_made)
        print(f"  [SUCCESS] Migrated ({changes_str})")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to save: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate annotated.csv files to current format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all
  %(prog)s --dir gen_1_experiments
  %(prog)s --file cf_outputs/experiment/annotated.csv
  %(prog)s --all --dry-run  # Preview without changes
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="Migrate all annotated.csv in cf_outputs"
    )
    parser.add_argument(
        "--dir", type=str, help="Migrate all annotated.csv in a specific directory"
    )
    parser.add_argument("--file", type=str, help="Migrate a single annotated.csv file")
    parser.add_argument(
        "--base-path",
        type=str,
        default="cf_outputs",
        help="Base path for cf_outputs directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without modifying files",
    )

    args = parser.parse_args()

    if not (args.all or args.dir or args.file):
        parser.error("Please specify --all, --dir, or --file")

    # Resolve base path
    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist")
        return 1

    # Find files
    files_to_process = []

    if args.file:
        file_path = Path(args.file)
        if file_path.is_absolute() or str(file_path).startswith(str(base_path)):
            files_to_process.append(file_path)
        else:
            files_to_process.append(base_path / file_path)
    elif args.all:
        files_to_process = list(base_path.glob("**/annotated.csv"))
    elif args.dir:
        dir_path = base_path / args.dir
        if not dir_path.exists():
            print(f"Error: Directory '{dir_path}' does not exist")
            return 1
        files_to_process = list(dir_path.glob("**/annotated.csv"))

    if not files_to_process:
        print("No annotated.csv files found!")
        return 1

    # Exclude backup and failed files
    files_to_process = [
        f
        for f in files_to_process
        if ".backup" not in f.name and ".failed" not in f.name
    ]

    print(f"\nFound {len(files_to_process)} annotated.csv file(s)")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")
    else:
        print("Migration backups will be created as .migration_backup\n")

    # Process each file
    success_count = 0
    error_count = 0

    for csv_path in sorted(files_to_process):
        try:
            rel_path = csv_path.relative_to(base_path)
        except ValueError:
            rel_path = csv_path

        print(f"Processing: {rel_path}")

        if migrate_single_file(csv_path, dry_run=args.dry_run):
            success_count += 1
        else:
            error_count += 1

        print()

    # Summary
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("MIGRATION COMPLETE")
    print(f"Total files: {len(files_to_process)}")
    print(f"Migrated: {success_count}")
    print(f"Errors: {error_count}")

    if not args.dry_run and success_count > 0:
        print("\nMigration backups saved as .migration_backup")
        print("Validation recommended: python scripts/validate_annotated_format.py")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
