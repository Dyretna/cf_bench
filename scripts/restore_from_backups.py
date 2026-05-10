#!/usr/bin/env python3
"""
Restore annotated.csv files from their .backup versions.

This script safely restores annotated.csv files from their .backup copies,
moving the current (potentially corrupted) versions to .failed for inspection.

Usage:
    python scripts/restore_from_backups.py --all
    python scripts/restore_from_backups.py --dir gen_1_experiments
    python scripts/restore_from_backups.py --file cf_outputs/path/to/annotated.csv
"""

import argparse
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def restore_single_file(csv_path: Path, dry_run: bool = False) -> bool:
    """
    Restore a single annotated.csv from its backup.

    Args:
        csv_path: Path to annotated.csv file
        dry_run: Only report what would be done

    Returns:
        True if restored successfully, False otherwise
    """
    backup_path = csv_path.parent / f"{csv_path.name}.backup"

    if not backup_path.exists():
        print(f"  [SKIP] No backup found: {backup_path.name}")
        return False

    if not csv_path.exists():
        print(f"  [SKIP] Original file doesn't exist: {csv_path.name}")
        return False

    if dry_run:
        print(f"  [DRY-RUN] Would restore from {backup_path.name}")
        return True

    # Move current file to .failed for inspection
    failed_path = csv_path.parent / f"{csv_path.name}.failed"
    try:
        shutil.move(str(csv_path), str(failed_path))
        print(f"  [OK] Moved current to {failed_path.name}")
    except Exception as e:
        print(f"  [ERROR] Failed to move current file: {e}")
        return False

    # Restore from backup
    try:
        shutil.copy2(backup_path, csv_path)
        print(f"  [SUCCESS] Restored from {backup_path.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to restore from backup: {e}")
        # Try to restore the failed file
        try:
            shutil.move(str(failed_path), str(csv_path))
            print("  [ROLLBACK] Restored original file")
        except Exception:
            pass
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Restore annotated.csv files from .backup copies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all
  %(prog)s --dir gen_1_experiments
  %(prog)s --file cf_outputs/experiment/annotated.csv
  %(prog)s --all --dry-run  # Preview without changes
        """,
    )

    # Input options
    parser.add_argument(
        "--all", action="store_true", help="Restore all annotated.csv in cf_outputs"
    )
    parser.add_argument(
        "--dir", type=str, help="Restore all annotated.csv in a specific directory"
    )
    parser.add_argument("--file", type=str, help="Restore a single annotated.csv file")
    parser.add_argument(
        "--base-path",
        type=str,
        default="cf_outputs",
        help="Base path for cf_outputs directory",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without modifying files",
    )

    args = parser.parse_args()

    # Validate arguments
    if not (args.all or args.dir or args.file):
        parser.error("Please specify --all, --dir, or --file")

    # Resolve base path
    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist")
        return 1

    # Find files to restore
    files_to_restore = []

    if args.file:
        file_path = Path(args.file)
        # If path is absolute or already contains cf_outputs, use as-is
        if file_path.is_absolute() or str(file_path).startswith(str(base_path)):
            files_to_restore.append(file_path)
        else:
            # Otherwise, prepend base_path
            file_path = base_path / file_path
            files_to_restore.append(file_path)
    elif args.all:
        files_to_restore = list(base_path.glob("**/annotated.csv"))
    elif args.dir:
        dir_path = base_path / args.dir
        if not dir_path.exists():
            print(f"Error: Directory '{dir_path}' does not exist")
            return 1
        files_to_restore = list(dir_path.glob("**/annotated.csv"))

    if not files_to_restore:
        print("No annotated.csv files found!")
        return 1

    # Exclude backup and failed files
    files_to_restore = [
        f
        for f in files_to_restore
        if ".backup" not in f.name and ".failed" not in f.name
    ]

    print(f"\nFound {len(files_to_restore)} annotated.csv file(s)")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")
    else:
        print("WARNING: Current annotated.csv files will be moved to .failed\n")

    # Process each file
    success_count = 0
    skip_count = 0
    error_count = 0

    for csv_path in sorted(files_to_restore):
        # Get relative path for display
        try:
            rel_path = csv_path.relative_to(base_path)
        except ValueError:
            rel_path = csv_path

        print(f"Processing: {rel_path}")

        result = restore_single_file(csv_path, dry_run=args.dry_run)

        if result:
            success_count += 1
        elif "No backup found" in str(result) or not result:
            skip_count += 1
        else:
            error_count += 1

        print()

    # Summary
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("RESTORE COMPLETE")
    print(f"Total files: {len(files_to_restore)}")
    print(f"Restored: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")

    if not args.dry_run and success_count > 0:
        print("\nCurrent files saved as .failed for inspection")
        print("Backups remain unchanged in .backup files")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
