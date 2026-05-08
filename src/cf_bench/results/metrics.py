"""Performance metrics calculation and formatting."""

from typing import Dict, List

import gower
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


class PerformanceMetrics:
    """Calculate and format model performance metrics."""

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, any]:
        report = classification_report(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        return {"classification_report": report, "roc_auc": roc_auc}

    @staticmethod
    def compute_timing_metrics(cf_times: List[float]) -> Dict[str, float]:
        if not cf_times:
            return {
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
                "num_instances": 0,
            }

        return {
            "total_time": sum(cf_times),
            "avg_time": sum(cf_times) / len(cf_times),
            "min_time": min(cf_times),
            "max_time": max(cf_times),
            "num_instances": len(cf_times),
        }

    @staticmethod
    def format_timing_metrics(timing_metrics: Dict[str, float]) -> str:
        return (
            f"Total CF generation time: {timing_metrics['total_time']:.2f}s\n"
            f"Average time per instance: {timing_metrics['avg_time']:.2f}s\n"
            f"Min time: {timing_metrics['min_time']:.2f}s\n"
            f"Max time: {timing_metrics['max_time']:.2f}s\n"
            f"Number of instances: {timing_metrics['num_instances']}\n"
        )

    @staticmethod
    def add_gower_distance_column(
        annotated_batch: pd.DataFrame,
        feature_cols: List[str],
        query_index_col: str = "query_index",
        cf_id_col: str = "cf_id",
    ) -> pd.DataFrame:
        """Add gower_distance column to annotated_batch DataFrame.

        Gower distance measures similarity between original and counterfactual.
        Works with mixed data types (continuous, categorical, ordinal).
        Lower values = more similar = fewer/smaller changes.

        Sets distance to empty string for 'original' rows.
        Calculates distance for CF rows compared to their original instance.

        Args:
            annotated_batch: DataFrame from build_annotated_batch with 'original' and CF rows
            feature_cols: List of feature column names to use for distance calculation
            query_index_col: Column identifying each person (default: "query_index")
            cf_id_col: Column identifying counterfactuals (default: "cf_id")

        Returns:
            DataFrame with added 'gower_distance' column (new copy, original unchanged)
        """

        # Make a copy to avoid mutating the original
        result = annotated_batch.copy()

        # Build distances in a plain Python dict first, then assign once.
        # This avoids Pandas StringDtype rejecting float assignments on a column
        # that was initialized with "".
        gower_distances = {}  # row_index -> float distance or "" for originals

        # Default to "" for every row (original rows keep this value)
        for idx in result.index:
            gower_distances[idx] = ""

        def normalize_for_gower(df):
            """Cast to numpy-compatible dtypes. Gower cannot handle pd.StringDtype."""
            out = df.copy()
            for c in out.columns:
                numeric = pd.to_numeric(out[c], errors="coerce")
                if numeric.notna().all():
                    out[c] = numeric
                else:
                    out[c] = out[c].astype(object)
            return out

        for query_idx in result[query_index_col].unique():
            # Get all rows for this person
            person_rows = result[result[query_index_col] == query_idx]

            # Find original row
            original_mask = person_rows[cf_id_col] == "original"
            if not original_mask.any():
                continue

            original_row = person_rows[original_mask][feature_cols]

            # Get CF rows
            cf_mask = person_rows[cf_id_col] != "original"
            if not cf_mask.any():
                continue

            cf_rows = person_rows[cf_mask]
            cf_features = cf_rows[feature_cols].copy()

            # Fill empty strings with original values before distance calculation.
            # Empty strings represent unchanged features (masked for CSV readability),
            # but Gower must see the actual values to compute a meaningful distance.
            for col in feature_cols:
                empty_mask = cf_features[col] == ""
                if empty_mask.any():
                    original_val = original_row[col].iloc[0]
                    cf_features.loc[empty_mask, col] = original_val

            original_row = normalize_for_gower(original_row)
            cf_features = normalize_for_gower(cf_features)

            # Compute Gower distances
            # gower_matrix returns [original] x [cf1, cf2, ...] → take first row [0]
            distances = gower.gower_matrix(original_row, cf_features)[0]

            for idx, dist in zip(cf_rows.index, distances):
                gower_distances[idx] = round(float(dist), 4)

        # Assign as object-dtype Series to avoid StringDtype rejection of floats
        result["gower_distance"] = pd.Series(gower_distances, dtype=object)

        return result
