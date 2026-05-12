# Scripts

Utility scripts for managing the cf_bench project.

## Data File Management

### migrate_annotated_results.py

Standardizes annotated.csv files to current format with proper rounding and column ordering.

### Usage

```bash
# Migrate all files
python scripts/migrate_annotated_results.py --all

# Migrate specific directory
python scripts/migrate_annotated_results.py --dir gen_1_experiments

# Migrate single file
python scripts/migrate_annotated_results.py --file cf_outputs/experiment/annotated.csv

# Dry run (preview without changes)
python scripts/migrate_annotated_results.py --all --dry-run
```

### Operations

- **Add Nchanged**: Counts modified features per CF
- **Compute gower_distance**: Calculates similarity metric
- **Reorder columns**: Standardizes column order
- **Apply rounding**:
  - gower_distance: 4 decimals
  - cf_gen_time_sec: 2 decimals
  - risk columns: 4 decimals
  - BMI: 5 decimals
  - Ordinal features: integers (no .0)

### Safety

- Creates `.migration_backup` before modifying files
- Preserves original `.backup` files

---

## restore_from_backups.py

Restores annotated.csv files from `.backup` copies.

### Usage

```bash
# Restore all backups
python scripts/restore_from_backups.py --all

# Restore specific directory
python scripts/restore_from_backups.py --dir gen_1_experiments

# Restore single file
python scripts/restore_from_backups.py --file cf_outputs/experiment/annotated.csv
```

### Safety

- Moves current files to `.failed` before restoring
- Only processes files with existing `.backup` copies
- Backup files remain unchanged

---

## analyze_csv_differences.py

Analyzes annotated.csv files to detect format inconsistencies.

### Usage

```bash
# Analyze all files
python scripts/analyze_csv_differences.py --all

# Analyze specific directory
python scripts/analyze_csv_differences.py --dir predictors_vs_threshold

# Save detailed report
python scripts/analyze_csv_differences.py --all --output analysis_report.txt
```

### Checks

- Nchanged column presence and population
- gower_distance presence and population
- Column order compliance
- Data type consistency

---

## Workflow

1. **Backup** → Create `.backup` files before making changes
2. **Analyze** → Run `analyze_csv_differences.py` to identify issues
3. **Migrate** → Run `migrate_annotated_results.py` to standardize
4. **Validate** → Re-run analysis to confirm standardization
5. **Restore** (if needed) → Use `restore_from_backups.py` to revert
