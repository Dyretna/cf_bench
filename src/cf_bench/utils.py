import pandas as pd


def filter_valid_cfs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only valid counterfactuals (where valid == True).

    Preserves all original instances and only keeps CFs that meet the target risk.
    This is useful for analysis where you only want to examine successful CFs.

    Args:
        df: DataFrame with 'cf_id' and 'valid' columns

    Returns:
        DataFrame with original rows + valid CFs only, sorted by query_index
    """
    df = df.copy()

    # Split baseline and CF rows (support both "original" and "xin")
    df_xin = df[df["cf_id"].isin(["original", "xin"])]
    df_cf = df[~df["cf_id"].isin(["original", "xin"])]

    # Keep only valid CFs (handle both boolean and string "True")
    # Convert to boolean dtype
    df_cf["valid"] = df_cf["valid"].astype(bool)
    df_cf_valid = df_cf[df_cf["valid"]].copy()

    # Combine baseline + valid CFs
    result = pd.concat([df_xin, df_cf_valid], ignore_index=True)

    # Ensure original appears before CFs for each query_index
    result["_is_original"] = result["cf_id"].isin(["original", "xin"]).astype(int)
    result = result.sort_values(
        ["query_index", "_is_original"], ascending=[True, False]
    ).drop(columns="_is_original")

    return result


def select_one_cf_per_query(
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
        # Convert valid column to boolean dtype
        group = group.copy()
        group["valid"] = group["valid"].astype(bool)

        # Check if we have the "Expected" column for violation detection
        has_expected_col = "Expected" in group.columns

        # First try: valid AND no violations (if prefer_no_violations and column exists)
        if prefer_no_violations and has_expected_col:
            good_cfs = group[group["valid"] & (group["Expected"] == "")]
            if len(good_cfs) > 0:
                return good_cfs.sample(n=1, random_state=random_state)

        # Fallback: any valid CF
        valid_cfs = group[group["valid"]]
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
