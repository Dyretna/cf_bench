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

    # Find query_indices with no valid CFs
    query_indices_with_valid = set(df_cf_valid["query_index"].unique())
    all_query_indices = set(df_xin["query_index"].unique())
    query_indices_without_valid = all_query_indices - query_indices_with_valid

    # Create placeholder rows for queries without valid CFs
    placeholder_rows = []
    for qi in query_indices_without_valid:
        # Get the original row to extract risk values
        orig_row = df_xin[df_xin["query_index"] == qi].iloc[0]

        # Create minimal row with query_index, valid, and risk columns
        placeholder = pd.Series(index=df.columns, dtype=object)
        placeholder["query_index"] = qi
        placeholder["cf_id"] = ""
        placeholder["valid"] = "False"
        placeholder["risk_before"] = orig_row["risk_before"]
        placeholder["predicted_risk_after"] = orig_row["predicted_risk_after"]

        # Fill all other columns with empty string
        for col in placeholder.index:
            if col not in [
                "query_index",
                "cf_id",
                "valid",
                "risk_before",
                "predicted_risk_after",
            ]:
                placeholder[col] = ""
        placeholder_rows.append(placeholder)

    # Combine all parts
    result_parts = [df_xin, df_cf_valid]
    if placeholder_rows:
        result_parts.append(pd.DataFrame(placeholder_rows))
    result = pd.concat(result_parts, ignore_index=True)

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
