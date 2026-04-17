"""Data validation utilities (placeholder for future validation logic)."""

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..config import SystemConfig


class DataValidator:
    """Validate data consistency and quality."""

    def __init__(self, config: "SystemConfig"):
        self.config = config

    def validate_dtypes(self, df: pd.DataFrame, context: str = "") -> bool:
        for col in self.config.ordinal_features:
            if col in df.columns:
                if df[col].dtype not in ["object", "str"]:
                    raise ValueError(
                        f"{context}: Expected ordinal feature '{col}' to have "
                        f"string dtype, got {df[col].dtype}"
                    )

        for col in self.config.continuous_features:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(
                        f"{context}: Expected continuous feature '{col}' to have "
                        f"numeric dtype, got {df[col].dtype}"
                    )

        return True

    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        required_cols = set(self.config.feature_cols)
        if self.config.target in df.columns:
            required_cols.add(self.config.target)

        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return True
