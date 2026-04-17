import pandas as pd

from .dice_adapters import BaseRiskEvaluator


def annotate_all(
    risk_evaluator: "BaseRiskEvaluator",
    cf_results: list,  # Changed: now a list of CounterfactualExamples
    query_df: pd.DataFrame,
) -> list[pd.DataFrame]:
    """
    Annotate all counterfactual batches.

    Parameters
    ----------
    risk_evaluator : RiskEvaluator
        Evaluator used to compute risk annotations.
    cf_results : list[dice_ml.CounterfactualExamples]
        List of counterfactual generation outputs (one per query instance).
    query_df : pandas.DataFrame
        Original query instances.

    Returns
    -------
    list[pandas.DataFrame]
        Annotated counterfactuals per query instance.
    """
    annotated_batches = []

    # Iterate over the list of CounterfactualExplanations objects
    for i, cf in enumerate(cf_results):
        qi = query_df.iloc[[i]]
        # Access cf_examples_list[0] since we generate one query at a time
        cf_df = cf.cf_examples_list[0].final_cfs_df
        annotated_batches.append(risk_evaluator.annotate(qi, cf_df))

    return annotated_batches


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
        orig_row["risk_before"] = risk_before
        orig_row["target_risk"] = target_risk
        orig_row[target] = outcome_value

        rows.append(orig_row)

        # --- CF ROWS ---
        for j, row in cf_risk.iterrows():
            cf_row = row.to_dict()
            cf_row["query_index"] = idx
            cf_row["cf_id"] = f"cf_{j + 1}"

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
