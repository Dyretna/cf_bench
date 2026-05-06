# =============================================================================
# assembler.py — Assemble and enrich annotated CF results
# =============================================================================
#
# PURPOSE:
#   Encapsulates all the logic for building the final annotated results table
#   from raw counterfactual outputs. This includes:
#     - Building the base annotated batch structure
#     - Adding CF generation timing data
#     - Computing metrics (Gower distance, nchanged)
#     - Reordering columns for readability
#
# WHY SEPARATE MODULE?
#   Keeps the main runner clean and focused on pipeline orchestration.
#   All "result assembly" concerns live here.
# =============================================================================

import pandas as pd

from .metrics import PerformanceMetrics


def assemble_annotated_results(
    query_df: pd.DataFrame,
    all_annotated: list,
    cf_times: list[float],
    target: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Assemble and enrich the final annotated results table.

    This function orchestrates all the steps needed to transform raw CF outputs
    into the final CSV structure with all metrics and proper formatting.

    Steps:
        1. Build base annotated batch (original + CF rows)
        2. Add CF generation timing
        3. Add Gower distance metric
        4. Add nchanged metric (count of modified features)
        5. Reorder columns for readability

    Args:
        query_df: Query instances (test data)
        all_annotated: List of annotated CF results from risk evaluator
        cf_times: List of generation times per instance (seconds)
        target: Target column name
        feature_cols: List of feature column names

    Returns:
        Fully assembled and enriched annotated results DataFrame
    """
    # 1. Build base structure
    annotated_batch = build_annotated_batch(
        query_instances=query_df,
        all_annotated=all_annotated,
        target=target,
    )

    # 2. Add CF generation timing
    annotated_batch = add_cf_generation_times(
        annotated_batch,
        query_df.index,
        cf_times,
    )

    # 3. Add Gower distance (similarity metric between original and CF)
    annotated_batch = PerformanceMetrics.add_gower_distance_column(
        annotated_batch=annotated_batch,
        feature_cols=feature_cols,
    )

    # 4. Add nchanged (count of modified features per CF)
    annotated_batch = add_nchanged_column(annotated_batch, feature_cols)

    # 5. Reorder columns for readability
    annotated_batch = reorder_annotated_columns(annotated_batch, feature_cols)

    return annotated_batch


def build_annotated_batch(query_instances, all_annotated, target):
    """
    Build a batch-level annotated DataFrame containing:
    - the original instance
    - all counterfactual rows
    - risk values
    - metadata (query_index, cf_id)
    """

    rows = []

    for i, (idx, original_row) in enumerate(
        zip(query_instances.index, query_instances.to_dict("records"))
    ):
        # Original instance as DataFrame
        original_df = pd.DataFrame([original_row])
        cf_risk = all_annotated[i]

        # Risk values (same for all CFs) - extract scalar values
        risk_before = cf_risk["risk_before"].iloc[0]
        target_risk = cf_risk["target_risk"].iloc[0]

        # Ensure we have scalar values, not Series
        if isinstance(risk_before, pd.Series):
            risk_before = risk_before.iloc[0]
        if isinstance(target_risk, pd.Series):
            target_risk = target_risk.iloc[0]

        # Outcome value
        outcome_value = original_df[target].iloc[0] if target in original_df else None

        # --- ORIGINAL ROW ---
        orig_row = {col: None for col in cf_risk.columns}

        for col, val in original_row.items():
            orig_row[col] = val

        orig_row["query_index"] = idx
        orig_row["cf_id"] = "original"
        orig_row["risk_before"] = (
            ""  # Original doesn't have a "before" (it IS the baseline)
        )
        orig_row["target_risk"] = target_risk
        orig_row[target] = outcome_value

        rows.append(orig_row)

        # --- CF ROWS ---
        # Use enumerate to get sequential numbering regardless of DataFrame index
        for j, (_, row) in enumerate(cf_risk.iterrows(), start=1):
            cf_row = row.to_dict()
            cf_row["query_index"] = idx
            cf_row["cf_id"] = f"cf_{j}"

            # clear (empty string) all values that's not changing
            for col in original_row.keys():
                if col in cf_row:
                    if cf_row[col] == original_row[col]:
                        cf_row[col] = ""

            rows.append(cf_row)

    annotated_batch = pd.DataFrame(rows)

    # Move metadata columns first
    first_cols = ["query_index", "cf_id"]
    other_cols = [c for c in annotated_batch.columns if c not in first_cols]
    annotated_batch = annotated_batch[first_cols + other_cols]

    return annotated_batch


def add_cf_generation_times(
    annotated_batch: pd.DataFrame,
    query_indices: pd.Index,
    cf_times: list[float],
) -> pd.DataFrame:
    """
    Add CF generation time to the annotated batch.

    Only 'original' rows get the timing value (one time per query instance).
    CF rows inherit None for this column.

    Args:
        annotated_batch: The annotated results dataframe
        query_indices: Index values from the original query_df
        cf_times: List of generation times (seconds) per instance

    Returns:
        DataFrame with cf_gen_time_sec column added
    """
    time_mapping = {idx: round(time, 2) for idx, time in zip(query_indices, cf_times)}

    annotated_batch["cf_gen_time_sec"] = annotated_batch.apply(
        lambda row: time_mapping.get(row["query_index"])
        if row["cf_id"] == "original"
        else None,
        axis=1,
    )

    return annotated_batch


def add_nchanged_column(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Add 'Nchanged' column counting how many features changed per CF."""
    df = df.copy()

    # Calculate Nchanged only for CF rows (count non-empty feature values)
    mask_cf = df["cf_id"] != "original"
    df.loc[mask_cf, "Nchanged"] = (df.loc[mask_cf, feature_cols] != "").sum(axis=1)

    # Convert to string and set original rows to empty string
    df["Nchanged"] = df["Nchanged"].astype("string")
    df.loc[df["cf_id"] == "original", "Nchanged"] = ""

    return df


def reorder_annotated_columns(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Reorder columns in the annotated batch for better readability."""

    order = (
        [
            "query_index",
            "cf_id",
        ]
        + feature_cols
        + [
            "cf_gen_time_sec",
            "gower_distance",
            "Nchanged",
            "valid",
            "risk_before",
            "predicted_risk_after",
            "target_risk",
        ]
    )

    # Only include columns that exist in the dataframe
    order = [col for col in order if col in df.columns]

    # Add any remaining columns at the end
    remaining = [col for col in df.columns if col not in order]
    order.extend(remaining)

    return df[order]
