"""Data transformation utilities for scaling and dtype conversion."""

from typing import TYPE_CHECKING, Optional, Tuple

import joblib
import pandas as pd

if TYPE_CHECKING:
    from ..config import SystemConfig


class FeatureScaler:
    """
    Handle feature scaling for models (typically Keras models).

    Provides forward and inverse transformations, handling dtype conversions
    for ordinal features when necessary.
    """

    def __init__(self, scaler, config: "SystemConfig"):
        """
        Initialize feature scaler.

        Parameters
        ----------
        scaler : sklearn scaler
            Fitted scaler object (e.g., StandardScaler, MinMaxScaler)
        config : SystemConfig
            Configuration containing feature metadata
        """
        self.scaler = scaler
        self.config = config

    def transform(
        self, df: pd.DataFrame, convert_ordinals: bool = True
    ) -> pd.DataFrame:
        """Scale features in DataFrame."""

        df_scaled = df.copy()

        # Convert string ordinals to numeric for scaling if needed
        if convert_ordinals:
            for col in self.config.ordinal_features:
                if col in df_scaled.columns and df_scaled[col].dtype == "object":
                    df_scaled[col] = pd.to_numeric(df_scaled[col])

        # Scale feature columns
        df_scaled[self.config.feature_cols] = self.scaler.transform(
            df_scaled[self.config.feature_cols]
        )

        return df_scaled

    def inverse_transform(
        self, df: pd.DataFrame, feature_cols: Optional[list] = None
    ) -> pd.DataFrame:
        """Inverse-transform feature columns back to raw space."""

        if feature_cols is None:
            feature_cols = self.config.feature_cols

        df_copy = df.copy()

        # Clean up any invalid values and ensure numeric dtypes
        df_copy[feature_cols] = df_copy[feature_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        # Inverse transform
        df_copy[feature_cols] = self.scaler.inverse_transform(df_copy[feature_cols])

        return df_copy


class DtypeConverter:
    """
    Convert between DiCE string ordinals and numeric representations.

    DiCE requires ordinal features as strings, while XGBoost models require
    numeric types. This class handles bidirectional conversion.
    """

    def __init__(self, config: "SystemConfig"):
        """
        Initialize dtype converter.

        Parameters
        ----------
        config : SystemConfig
            Configuration containing ordinal feature metadata
        """
        self.config = config

    def to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert ordinal features from strings to numeric."""
        df_converted = df.copy()

        for col in self.config.ordinal_features:
            if col in df_converted.columns and df_converted[col].dtype == "object":
                df_converted[col] = pd.to_numeric(df_converted[col])

        return df_converted

    def to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert ordinal features from numeric to strings."""
        df_converted = df.copy()

        for col in self.config.ordinal_features:
            if col in df_converted.columns:
                if pd.api.types.is_numeric_dtype(df_converted[col]):
                    # Round float64 values to nearest int, then convert to string
                    df_converted[col] = (
                        df_converted[col].round().astype(int).astype(str)
                    )

        return df_converted


class QueryInstancePreparer:
    """
    Prepare query instances for DiCE.

    Handles complex dtype alignment between model output (numeric) and DiCE input
    (string ordinals). Ensures query instances match the dtype of training data
    used by DiCE.
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        self.dtype_converter = DtypeConverter(config)

    def prepare(
        self, model_input_df: pd.DataFrame, training_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare query instances for DiCE from model input data."""

        # Drop target column to get query instances
        query_df = model_input_df.drop(columns=[self.config.target])

        # Convert ordinal features from numeric to string
        query_df = self.dtype_converter.to_string(query_df)

        # Ensure dtypes match training data exactly
        training_dtypes = training_df.drop(
            columns=[self.config.target]
        ).dtypes.to_dict()
        query_df = query_df.astype(training_dtypes)

        return query_df


def scale_data_if_keras(
    df: pd.DataFrame, config: "SystemConfig", scaler_path: str, is_keras: bool
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Optionally scale data if the underlying model is a Keras model.

    Note: Scaling is applied to the numeric representation of features.
    For ordinal features loaded as strings, we first convert to numeric,
    scale, then can convert back to strings for DiCE compatibility if needed.
    """

    if not is_keras:
        return df, None

    scaler = joblib.load(scaler_path)
    feature_scaler = FeatureScaler(scaler, config)
    df_scaled = feature_scaler.transform(df, convert_ordinals=True)

    return df_scaled, scaler
