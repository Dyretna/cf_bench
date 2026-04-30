import pandas as pd


class DiceRecommender:
    """Format counterfactual recommendations from batch dataframes."""

    def __init__(self, target: str = "hltprhc", feature_cols: list[str] | None = None):
        self.target = target
        self.feature_cols = feature_cols or [
            "etfruit",
            "eatveg",
            "cgtsmok",
            "alcfreq",
            "slprl",
            "paccnois",
            "bmi",
            "dosprt",
        ]

    def get_recommendations_for_query(
        self, batch_df: pd.DataFrame, query_index: int
    ) -> list[dict]:
        """
        Extract recommendations for a single query_index from batch dataframe.

        Parameters
        ----------
        batch_df : pd.DataFrame
            Batch dataframe with columns: query_index, cf_id, features, etc.
            Should contain both 'xin' (original) and cf rows for the query_index.
        query_index : int
            Which query_index to extract recommendations for.

        Returns
        -------
        list[dict]
            List of dicts with cf_id, risk info, and feature changes.
        """
        # Filter to this query_index
        query_data = batch_df[batch_df["query_index"] == query_index].copy()

        if query_data.empty:
            return []

        # Get the xin (original) row
        xin_row = query_data[query_data["cf_id"] == "xin"]
        if xin_row.empty:
            return []

        xin_row = xin_row.iloc[0]

        # Get CF rows
        cf_rows = query_data[query_data["cf_id"] != "xin"]

        results = []

        # Process each CF
        for _, cf_row in cf_rows.iterrows():
            changes = []
            for col in self.feature_cols:
                orig = xin_row[col]
                new = cf_row[col]

                # Skip if either is empty/nan
                if pd.isna(orig) or pd.isna(new) or orig == "" or new == "":
                    continue

                # Convert to comparable format
                if isinstance(orig, str):
                    orig = orig.strip()
                if isinstance(new, str):
                    new = new.strip()

                if orig != new:
                    changes.append((col, orig, new))

            results.append(
                {
                    "query_index": query_index,
                    "cf_id": cf_row.get("cf_id"),
                    "risk_before": xin_row.get("risk_before"),
                    "target_risk": None,  # Not in batch format
                    "predicted_risk_after": cf_row.get("predicted_risk_after"),
                    "valid": cf_row.get("valid"),
                    "changes": changes,
                }
            )

        return results

    def format_recommendations(
        self, recs: list[dict], xin_row: pd.Series | None = None
    ) -> str:
        """
        Format counterfactual recommendations into human-readable text.

        Parameters
        ----------
        recs : list[dict]
            List of recommendation dicts from get_recommendations_for_query()
        xin_row : pd.Series, optional
            The original instance row (for displaying full features).
            If None, only changes are shown.

        Returns
        -------
        str
            Formatted multi-line string with CF recommendations
        """
        if not recs:
            return "No recommendations available."

        qidx = recs[0].get("query_index")
        lines = []

        # Header
        lines.append(f"Task / Target: {self.target}")
        lines.append(f"Query instance index: {qidx}\n")

        # Original instance (if provided)
        if xin_row is not None:
            lines.append("Original instance:")
            for feat in self.feature_cols:
                val = xin_row.get(feat, "")
                lines.append(f"  {feat}: {val}")
            lines.append("")

        # Risk info
        risk_before = recs[0].get("risk_before", "")
        if risk_before and risk_before != "":
            lines.append(f"Original predicted risk: {risk_before}")
        lines.append("")

        lines.append("=== Counterfactuals ===\n")

        # CFs
        for r in recs:
            cf_id = r.get("cf_id")
            risk = r.get("predicted_risk_after")
            valid = r.get("valid")
            changes = r.get("changes", [])

            lines.append(f"--- {cf_id} ---")
            if risk and risk != "":
                lines.append(f"Predicted risk: {risk}")
            lines.append(f"Valid: {valid}")

            if not changes:
                lines.append("No feature changes.\n")
                continue

            lines.append("Changes:")
            for feat, old, new in changes:
                lines.append(f"  {feat}: {old} → {new}")

            lines.append("")

        return "\n".join(lines)

    def format_query(self, batch_df: pd.DataFrame, query_index: int) -> str:
        """
        Get formatted recommendations for a single query from batch dataframe.

        Convenience method combining get_recommendations_for_query() and format_recommendations().

        Parameters
        ----------
        batch_df : pd.DataFrame
            Batch dataframe with query_index, cf_id, features, etc.
        query_index : int
            Which query to format

        Returns
        -------
        str
            Formatted recommendation text
        """
        print(f"\n=== Query index '{query_index}' ===")
        recs = self.get_recommendations_for_query(batch_df, query_index)

        # Get xin row for display
        xin_row = batch_df[
            (batch_df["query_index"] == query_index) & (batch_df["cf_id"] == "xin")
        ]
        xin_row = xin_row.iloc[0] if not xin_row.empty else None

        return self.format_recommendations(recs, xin_row)
