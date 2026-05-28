"""Data filtering utilities for CF benchmark results."""

import pandas as pd


def filter_valid_cfs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only valid counterfactuals (where valid == True).

    Preserves all original instances and only keeps CFs that meet the target risk.
    For queries with no valid CFs, adds a placeholder row with valid=False.
    This is useful for analysis where you only want to examine successful CFs.

    Args:
        df: DataFrame with 'cf_id' and 'valid' columns

    Returns:
        DataFrame with original rows + valid CFs (or False placeholder), sorted by query_index
    """
    df = df.copy()

    # Split baseline and CF rows (support both "original" and "xin")
    df_xin = df[df["cf_id"].isin(["original", "xin"])].copy()
    df_cf = df[~df["cf_id"].isin(["original", "xin"])].copy()

    # Keep only valid CFs (handle both boolean True and string "True")
    # Create a boolean mask that works for both types
    is_valid = df_cf["valid"].isin([True, "True"])
    df_cf_valid = df_cf[is_valid].copy()

    # Find query_indices with no valid CFs and get first invalid CF instead
    query_indices_with_valid = set(df_cf_valid["query_index"].unique())
    all_query_indices = set(df_cf["query_index"].unique())
    query_indices_without_valid = all_query_indices - query_indices_with_valid

    # For queries without valid CFs, take the first invalid CF
    first_invalid_cfs = []
    for qi in query_indices_without_valid:
        first_invalid = df_cf[df_cf["query_index"] == qi].iloc[0:1]
        first_invalid_cfs.append(first_invalid)

    # Combine all parts
    result_parts = [df_xin, df_cf_valid]
    if first_invalid_cfs:
        result_parts.append(pd.concat(first_invalid_cfs, ignore_index=True))
    result = pd.concat(result_parts, ignore_index=True)

    # Ensure original appears before CFs for each query_index
    result["_is_original"] = result["cf_id"].isin(["original", "xin"]).astype(int)
    result = result.sort_values(
        ["query_index", "_is_original"], ascending=[True, False]
    ).drop(columns="_is_original")

    return result


def select_one_cf_per_query_legacy(
    df: pd.DataFrame, prefer_no_violations: bool = True, random_state: int = 42
) -> pd.DataFrame:
    """
    Select one counterfactual per query instance.

    Keeps all original instances and selects one CF per query_index.
    Selection strategy (in priority order):
    1. Valid CF without violations (if prefer_no_violations=True and "Expected" column exists)
    2. Any valid CF
    3. First available CF

    Args:
        df: DataFrame with 'query_index', 'cf_id', and 'valid' columns
        prefer_no_violations: If True, prioritize CFs without violations (requires "Expected" column)
        random_state: Random seed for reproducible sampling

    Returns:
        DataFrame with original rows + one CF per query_index
    """
    df = df.copy()

    # Split baseline and CF rows (support both "original" and "xin")
    df_xin = df[df["cf_id"].isin(["original", "xin"])]
    df_cf = df[~df["cf_id"].isin(["original", "xin"])]

    def select_best_cf(group):
        """Select one CF per query, preferring those without violations."""
        # Create boolean mask that works for both boolean True and string "True"
        is_valid = group["valid"].isin([True, "True"])

        # Check if we have the "Expected" column for violation detection
        has_expected_col = "Expected" in group.columns

        # First try: valid AND no violations (if prefer_no_violations and column exists)
        if prefer_no_violations and has_expected_col:
            good_cfs = group[is_valid & (group["Expected"] == "")]
            if len(good_cfs) > 0:
                return good_cfs.sample(n=1, random_state=random_state)

        # Fallback: any valid CF
        valid_cfs = group[is_valid]
        if len(valid_cfs) > 0:
            return valid_cfs.sample(n=1, random_state=random_state)

        # Last resort: first CF
        return group.iloc[[0]]

    # Select one CF per query_index
    df_cf_selected = (
        df_cf.groupby("query_index", group_keys=False)
        .apply(select_best_cf)
        .reset_index(drop=True)
    )

    # Combine baseline + selected CF
    result = pd.concat([df_xin, df_cf_selected], ignore_index=True)

    # Ensure original appears before CF for each query_index
    result["_is_original"] = result["cf_id"].isin(["original", "xin"]).astype(int)
    result = result.sort_values(
        ["query_index", "_is_original"], ascending=[True, False]
    ).drop(columns="_is_original")

    return result


# -----------------------------------------------------------------------------
# NEW Notebook variants.
# added select_random_cfs, since it is what we used in old noteboks
# select_best_cfs now takes the CF with lower GOwer, instead of a random CF
# -----------------------------------------------------------------------------

# ---------------------------------------------------------
# LEGACY FUNCTION — RANDOM CF SELECTOR
# ---------------------------------------------------------


def select_random_cf(group):
    """
    LEGACY FUNCTION.
    Select one random CF that is valid and has no violations (Expected == "").
    This function is kept for reproducibility of earlier experiments.
    New analyses should use select_best_cf() instead.
    """
    group = group.copy()
    is_valid = group["valid"].isin([True, "True"])

    valid_cfs = group[is_valid]
    if len(valid_cfs) > 0:
        return valid_cfs.sample(n=1, random_state=42)

    return group.iloc[[0]]


# ---------------------------------------------------------
# NEW FUNCTION — LOWEST GOWER SELECTOR
# ---------------------------------------------------------


def select_gower_cf(group):
    """
    Select the valid CF with the lowest Gower distance, preferring no violations.
    This is the recommended method for all new analyses.
    """
    group = group.copy()
    group["gower_distance"] = pd.to_numeric(group["gower_distance"], errors="coerce")

    # Any valid CF
    is_valid = group["valid"].isin([True, "True"])
    valid = group[is_valid]
    if len(valid) > 0:
        return valid.nsmallest(1, "gower_distance")

    # Last resort
    return group.iloc[[0]]


# ---------------------------------------------------------
# FULL PIPELINE — PREPARE + SELECT + MERGE + SORT
# ---------------------------------------------------------


def select_one_cf_per_query(df: pd.DataFrame, selector):
    """
    Full pipeline:
    - Convert Gower to numeric
    - Split xin vs CF rows
    - Select exactly one CF per query_index using chosen selector
    - Merge xin + selected CF
    - Sort so xin appears first

    selector options:
        "gower"   -> select_gower_cf (lowest Gower)
        "random" -> select_random_cf (legacy)
    """
    df = df.copy()
    df["gower_distance"] = pd.to_numeric(df["gower_distance"], errors="coerce")

    # Split baseline and CF rows
    df_xin = df[df["cf_id"] == "xin"]
    df_cf = df[df["cf_id"] != "xin"]

    # Choose selector function
    if selector == "random":
        func = select_random_cf
    else:
        func = select_gower_cf

    # Apply selection
    df_cf_selected = (
        df_cf.groupby("query_index", group_keys=False)
        .apply(func)
        .reset_index(drop=True)
    )

    # Combine xin + selected CF
    out = pd.concat([df_xin, df_cf_selected], ignore_index=True)

    # Ensure xin appears first
    out["is_xin"] = (out["cf_id"] == "xin").astype(int)
    out = out.sort_values(["query_index", "is_xin"], ascending=[True, False]).drop(
        columns="is_xin"
    )

    # Convert Gower to string and fill NaN
    out["gower_distance"] = out["gower_distance"].astype("object").fillna("")

    return out
