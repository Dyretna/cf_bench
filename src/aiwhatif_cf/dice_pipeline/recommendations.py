import pandas as pd


class DiceRecommender:
    def __init__(self, feature_cols: list[str], target: str):
        self.feature_cols = feature_cols
        self.target = target

    def get_recommendations(
        self,
        query_instances: pd.DataFrame,
        annoted_counterfactuals: pd.DataFrame,
    ) -> list[dict]:
        """
        Compute human-readable feature changes for each counterfactual instance.

        This function compares a single original instance against a set of
        counterfactual rows and extracts all feature-level changes. It does not
        perform any printing or formatting; instead, it returns a structured list
        that can be used by CLI tools, notebooks, or reporting layers.

        Parameters
        ----------
        query_instances : pd.Dataframe
            A single-row representation of the original individual from which
            counterfactuals were generated.

        counterfactuals : pd.DataFrame
            A DataFrame containing one or more counterfactual instances. Each row
            is compared against the original instance to identify feature changes.

        feature_cols : list[str]
            The set of feature columns to compare between the original instance
            and each counterfactual row.

        Returns
        -------
        list of dict
            A list where each element corresponds to one counterfactual and
            contains:
                - "cf_id": Optional identifier for the counterfactual (if present)
                - "predicted_risk": The predicted risk for the CF (if present)
                - "meets_target_risk": Whether the CF meets the risk target (if present)
                - "changes": A list of (feature, original_value, cf_value) tuples
                describing all feature-level modifications.

            Example element:
            {
                "cf_id": "CF3",
                "predicted_risk": 0.12,
                "meets_target_risk": True,
                "changes": [
                    ("bmi", 31.4, 27.8),
                    ("smoking", 1, 0),
                ]
            }
        """
        results = []

        for _, row in annoted_counterfactuals.iterrows():
            changes = []
            for col in self.feature_cols:
                orig = query_instances[col].iloc[0]
                new = row[col]
                if pd.isna(orig) or pd.isna(new):
                    continue
                if orig != new:
                    changes.append((col, orig, new))

            results.append(
                {
                    "query_index": row.get("query_index"),
                    "cf_id": row.get("cf_id"),
                    "original_risk": row.get("original_risk"),
                    "target_risk": row.get("target_risk"),
                    "predicted_risk": row.get("predicted_risk"),
                    "meets_target_risk": row.get("meets_target_risk"),
                    "changes": changes,
                }
            )

        return results

    def format_recommendations(
        self, query_instances: pd.DataFrame, recs: list[dict], true_outcome: int | str
    ) -> str:
        """
        Format original instance information and counterfactual recommendations
        into a single human-readable string.

        Parameters
        ----------
        query_instances : pd.DataFrame
            A single-row DataFrame containing the original feature values.
        recs : list[dict]
            A list of counterfactual recommendation dictionaries. Each dict must
            contain:
                - "query_index": int
                - "cf_id": str
                - "predicted_risk": float or None
                - "meets_target_risk": bool or None
                - "changes": list of (feature, old_value, new_value)
                - "original_risk": float (only present on the first element)
                - "target_risk": float (only present on the first element)
        true_outcome : int or str
            The true label of the original instance (e.g., 0 or 1).

        Returns
        -------
        str
            A formatted multi-line string containing:
                - metadata (task, filename, query index)
                - original instance features
                - original and target risk
                - per-counterfactual changes and risk info
        """

        if query_instances is None or query_instances.empty:
            return "No query instance provided."

        if not recs:
            return "No recommendations available."

        qidx = recs[0].get("query_index", None)

        lines: list[str] = []

        # --- Header ---
        lines.append(f"Task / Target: {self.target}")
        lines.append(f"Selected query instance (index {qidx}):\n")

        # Original instance
        lines.append(query_instances.to_string(index=False))
        lines.append("")

        # Original + target risk
        orig_risk = recs[0].get("original_risk")
        target_risk = recs[0].get("target_risk")

        lines.append(
            f"Original predicted risk (P({self.target}={true_outcome})): "
            f"{orig_risk:.4f}"
        )
        lines.append(f"Target-risk threshold: {target_risk:.4f}")
        lines.append("")
        lines.append("=== Changes per Counterfactual (vs Original) ===")
        lines.append("")

        # --- Counterfactuals ---
        for r in recs:
            cf_id = r.get("cf_id")
            risk = r.get("predicted_risk")
            meets = r.get("meets_target_risk")
            changes = r.get("changes", [])

            lines.append(f"--- {cf_id} ---")

            # Risk may be None for original row
            if risk is None:
                lines.append("Predicted risk: None")
            else:
                lines.append(f"Predicted risk: {risk:.4f}")

            lines.append(f"Meets target:   {meets}")

            if not changes:
                lines.append("No feature changes.\n")
                continue

            lines.append("Changes:")
            for feat, old, new in changes:
                lines.append(f"  - {feat}: {old:.3f} → {new:.3f}")

            lines.append("")

        return "\n".join(lines)
