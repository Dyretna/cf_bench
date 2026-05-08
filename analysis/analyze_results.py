"""
Analysis of counterfactual experiment results.

We have 5 experiments, each with a CSV file.
This script reads each CSV and computes the same set of metrics for each,
so we can compare them side by side.

HOW THE CSV IS STRUCTURED:
  - One row per person with cf_id = "original"  → their real data and risk
  - Many rows per person with cf_id = "cf_1"    → counterfactual suggestions
  - A blank cell (NaN) in a feature column means that feature was NOT changed
  - A non-blank cell means DiCE DID change that feature in this counterfactual
"""

import pandas as pd


# ── The 8 lifestyle features DiCE is allowed to modify ────────────────────
FEATURE_COLUMNS = [
    "etfruit",   # fruit consumption
    "eatveg",    # vegetable consumption
    "cgtsmok",   # smoking
    "alcfreq",   # alcohol frequency
    "slprl",     # sleep quality
    "paccnois",  # physical activity noise
    "bmi",       # body mass index
    "dosprt",    # sport frequency
]

# ── List of experiments to analyse ────────────────────────────────────────
# Each experiment is one CSV file from one pipeline run.
# "locked" = features that were supposed to be frozen in that experiment.
EXPERIMENTS = [
    {
        "name": "Sparsity 0.1",
        "csv": "cf_outputs/sparsity_tests/XGBoost_genetic_hltprhc_2026-04-30_sp0.1/genetic_annotated_hltprhc.csv",
        "locked": [],
    },
    {
        "name": "Sparsity 0.5",
        "csv": "cf_outputs/sparsity_tests/XGBoost_genetic_hltprhc_2026-04-30_sp0.5/genetic_annotated_hltprhc.csv",
        "locked": [],
    },
    {
        "name": "Sparsity 0.9",
        "csv": "cf_outputs/sparsity_tests/XGBoost_genetic_hltprhc_2026-04-30_sp0.9/genetic_annotated_hltprhc.csv",
        "locked": [],
    },
    {
        "name": "Lock: smoking",
        "csv": "cf_outputs/feature_locking_tests/XGBoost_genetic_hltprhc_2026-04-30_lock_smoking/genetic_annotated_hltprhc.csv",
        "locked": ["cgtsmok"],
    },
    {
        "name": "Lock: BMI + smoking",
        "csv": "cf_outputs/feature_locking_tests/XGBoost_genetic_hltprhc_2026-04-30_lock_bmi_smoking/genetic_annotated_hltprhc.csv",
        "locked": ["cgtsmok", "bmi"],
    },
]


# =============================================================================
# METRIC 1 — How many features does DiCE change per counterfactual?
# =============================================================================
def avg_features_changed(cf_rows):
    """
    For each CF row, count how many feature cells are NOT blank.
    A non-blank cell = DiCE changed that feature.
    Then return the average across all CF rows.

    Example for one row:
      etfruit=NaN  eatveg=NaN  cgtsmok=NaN  alcfreq=7.0  bmi=18.97  ...
      → notna() gives: False False False True True ...
      → sum gives: 2   (2 features were changed in this CF)
    """
    # notna() returns True where a cell has a value, False where it is blank
    has_value = cf_rows[FEATURE_COLUMNS].notna()

    # sum(axis=1) counts the Trues across each row (axis=1 means "per row")
    n_changed_per_row = has_value.sum(axis=1)

    # mean() gives the average across all CF rows
    return n_changed_per_row.mean()


# =============================================================================
# METRIC 2 — What percentage of counterfactuals are valid?
# =============================================================================
def validity_rate(cf_rows):
    """
    A CF is "valid" if it actually achieves the goal:
    predicted_risk_after < target_risk (the valid column is True).

    Returns the percentage of valid CFs out of all CFs.
    """
    n_total = len(cf_rows)
    n_valid = len(cf_rows[cf_rows["valid"] == True])

    if n_total == 0:
        return 0

    return n_valid / n_total * 100


# =============================================================================
# METRIC 3 — For how many people did DiCE find at least one valid CF?
# =============================================================================
def pct_people_solved(cf_rows, n_people):
    """
    Even if only 1 CF out of 20 is valid for a person, that person is "solved".
    We count how many people have at least 1 valid CF.
    """
    n_solved = 0

    # Loop over each person one by one
    for person_id in cf_rows["query_index"].unique():
        # Get all CF rows for this person
        this_person_cfs = cf_rows[cf_rows["query_index"] == person_id]

        # Check if any of their CFs is valid
        at_least_one_valid = (this_person_cfs["valid"] == True).any()

        if at_least_one_valid:
            n_solved += 1

    return n_solved / n_people * 100


# =============================================================================
# METRIC 4 — How much does risk drop in valid CFs?
# =============================================================================
def avg_risk_reduction(valid_cf_rows, original_rows):
    """
    For each valid CF:
      risk_before    = the person's original risk (from the original row)
      risk_after     = the predicted risk after applying this CF

    Reduction = (risk_before - risk_after) / risk_before × 100
    We average this across all valid CFs.
    """
    if len(valid_cf_rows) == 0:
        return None

    results = []

    for person_id in valid_cf_rows["query_index"].unique():
        # Original risk for this person
        original_row = original_rows[original_rows["query_index"] == person_id]
        risk_before = original_row["risk_before"].values[0]

        # All valid CFs for this person
        this_person_valid = valid_cf_rows[valid_cf_rows["query_index"] == person_id]

        for _, cf_row in this_person_valid.iterrows():
            risk_after = cf_row["predicted_risk_after"]
            reduction = (risk_before - risk_after) / risk_before * 100
            results.append(reduction)

    return sum(results) / len(results)


# =============================================================================
# METRIC 5 — Which features are most often changed in valid CFs?
# =============================================================================
def feature_frequency(valid_cf_rows):
    """
    For each of the 8 features, count in what percentage of valid CFs
    that feature was changed (i.e., was not blank).

    Returns a sorted list like: "bmi(100%), etfruit(38%), ..."
    """
    if len(valid_cf_rows) == 0:
        return "no valid CFs"

    n_valid = len(valid_cf_rows)
    parts = []

    for feature in FEATURE_COLUMNS:
        # Count how many valid CF rows have a non-blank value for this feature
        n_changed = valid_cf_rows[feature].notna().sum()
        pct = n_changed / n_valid * 100
        if pct > 0:
            parts.append((feature, pct))

    # Sort by frequency, most common first
    parts.sort(key=lambda x: x[1], reverse=True)

    return "  ".join(f"{feat}={pct:.0f}%" for feat, pct in parts)


# =============================================================================
# METRIC 6 — Did DiCE respect the locked features?
# =============================================================================
def check_locking(cf_rows, locked_features):
    """
    For each locked feature, count how many CF rows changed it anyway.
    Zero violations = the lock worked correctly.
    """
    if not locked_features:
        return "no locks"

    results = []
    for feature in locked_features:
        n_violations = cf_rows[feature].notna().sum()
        results.append(f"{feature}: {n_violations} violations")

    return "  |  ".join(results)


# =============================================================================
# MAIN — Read each experiment CSV and print a comparison table
# =============================================================================
def main():
    print()
    print("=" * 75)
    print("COUNTERFACTUAL EXPERIMENT RESULTS — COMPARISON TABLE")
    print("=" * 75)

    for experiment in EXPERIMENTS:
        print()
        print(f"--- {experiment['name']} ---")

        # Read the CSV file
        df = pd.read_csv(experiment["csv"])

        # Separate original rows from CF rows
        original_rows = df[df["cf_id"] == "original"]
        cf_rows       = df[df["cf_id"] != "original"]
        valid_cf_rows = cf_rows[cf_rows["valid"] == True]

        # Basic counts
        n_people    = len(original_rows)
        n_total_cfs = len(cf_rows)
        n_valid_cfs = len(valid_cf_rows)

        # Compute each metric
        mean_features    = avg_features_changed(cf_rows)
        rate_valid       = validity_rate(cf_rows)
        pct_solved       = pct_people_solved(cf_rows, n_people)
        risk_drop        = avg_risk_reduction(valid_cf_rows, original_rows)
        top_features     = feature_frequency(valid_cf_rows)
        avg_time         = original_rows["cf_gen_time_sec"].mean()
        locking_check    = check_locking(cf_rows, experiment["locked"])

        # Print results
        print(f"  People in test set          : {n_people}")
        print(f"  Total CFs generated         : {n_total_cfs}")
        print(f"  Valid CFs                   : {n_valid_cfs}  ({rate_valid:.1f}%)")
        print(f"  People with 1+ valid CF     : {pct_solved:.1f}%")
        print(f"  Avg features changed / CF   : {mean_features:.2f}  (out of 8 possible)")
        if risk_drop is not None:
            print(f"  Avg risk reduction (valid)  : {risk_drop:.1f}%")
        print(f"  Avg generation time / person: {avg_time:.2f} sec")
        print(f"  Most changed features       : {top_features}")
        print(f"  Locking check               : {locking_check}")

    print()
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
