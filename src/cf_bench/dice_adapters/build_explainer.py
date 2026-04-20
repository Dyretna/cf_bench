import logging

import dice_ml
import pandas as pd
from dice_ml.explainer_interfaces.dice_tensorflow2 import DiceTensorFlow2

from ..config import SystemConfig

logger = logging.getLogger(__name__)


class SanitizedModel:
    """
    A generic model wrapper that enforces consistent input dtypes.

    DiCE generates synthetic counterfactual samples with string dtypes for
    ordinal/categorical features (as required by DiCE's internal validation).
    However, models like XGBoost require strict numeric dtypes and will fail
    if given strings or mixed types.

    This wrapper bridges the gap:
    - Accepts DiCE output (strings for ordinals)
    - Converts to numeric dtypes (what the model expects)
    - Forwards to the underlying model for prediction
    - Works for any sklearn-compatible model without special-casing

    The conversion strategy:
    1. Store the numeric dtypes from model_input_df (XGBoost's expected input)
    2. When DiCE calls predict, convert string ordinals -> numeric
    3. Cast all features to match the stored dtypes
    4. Forward to the actual model
    """

    def __init__(self, model, train_df):
        """
        Initialize the sanitized model wrapper.

        Parameters
        ----------
        model : sklearn-compatible model
            The trained predictive model (e.g., XGBoost, RandomForest)
        train_df : pd.DataFrame
            Training data with NUMERIC dtypes (as expected by the model).
            This is model_input_df, NOT the DiCE training data.
        """
        self.model = model
        # Store numeric dtypes that the model expects
        self.train_dtypes = train_df.dtypes.to_dict()

    def _sanitize(self, df) -> pd.DataFrame:
        """Convert DiCE output (strings for ordinals) to numeric dtypes."""
        df = df.copy()

        logger.debug("SanitizedModel._sanitize() called")
        logger.debug(f"Input dtypes: {df.dtypes.to_dict()}")
        logger.debug(f"Target dtypes: {self.train_dtypes}")
        logger.debug(f"Input shape: {df.shape}")

        for col, target_dtype in self.train_dtypes.items():
            if col not in df.columns:
                logger.debug(f"  SKIP {col}: not in df.columns")
                continue

            current_dtype = df[col].dtype
            logger.debug(f"  {col}: {current_dtype} -> {target_dtype}")

            # Convert string/object to numeric ALWAYS (don't skip)
            if current_dtype == "object" or pd.api.types.is_string_dtype(current_dtype):
                logger.debug(f"    Converting {col} from object/string to numeric")
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.debug(f"    After pd.to_numeric: {df[col].dtype}")

            # Cast to target dtype
            try:
                df[col] = df[col].astype(target_dtype)
                logger.debug(f"    After astype: {df[col].dtype}")
            except Exception as e:
                logger.warning(f"    Error casting {col}: {e}, using coerce fallback")
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(target_dtype)

        logger.debug(f"Output dtypes: {df.dtypes.to_dict()}")
        return df

    def predict_proba(self, df):
        """
        Sanitize input and forward to model.predict_proba.

        _sanitize() handles all dtype conversions (strings -> numeric).
        """
        df_sanitized = self._sanitize(df)
        return self.model.predict_proba(df_sanitized)

    def predict(self, df):
        """
        Sanitize input and forward to model.predict.

        _sanitize() handles all dtype conversions (strings -> numeric).
        """
        df_sanitized = self._sanitize(df)
        return self.model.predict(df_sanitized)


# ----------------------------------------------------------------------
#  Build DiCE explainer for both sklearn and TF2 backends
# ----------------------------------------------------------------------


def build_explainer(
    config: SystemConfig, predictor_model, model_input_df, training_df, explainer_method
):
    """
    Build and return a DiCE explainer instance.

    CRITICAL DTYPE HANDLING:
    - training_df: ordinals are STRINGS (loaded by _load_data for DiCE)
    - model_input_df: ordinals are NUMERIC (for XGBoost/model prediction)
    - SanitizedModel bridges this gap: accepts strings, outputs numeric

    Parameters
    ----------
    config : SystemConfig
        Configuration object containing feature metadata, backend, model type,
        and default explainer method.
    predictor_model : Any
        The trained predictive model to wrap inside the DiCE Model object.
    model_input_df : pd.DataFrame
        The dataset with NUMERIC dtypes (as expected by the model).
        Used to initialize SanitizedModel.
    training_df : pd.DataFrame
        The dataset with STRING dtypes for ordinals (as expected by DiCE).
        Used to construct the DiCE Data object.
    explainer_method : str
        Defines the DiCE explainer type,
        e.g. 'random', 'genetic' or 'gradientdescent'

    Returns
    -------
    dice_ml.Dice
        A fully constructed DiCE explainer ready to generate counterfactuals.
    """

    # -----------------------------
    # Build Data Interface
    # -----------------------------
    # training_df has string dtypes for ordinals (from _load_data)
    dice_data = dice_ml.Data(
        dataframe=training_df,
        continuous_features=config.continuous_features,
        outcome_name=config.target,
    )

    # -----------------------------
    # SKLEARN BACKEND
    # -----------------------------
    if config.backend == "sklearn":
        # Wrap model to convert string ordinals → numeric for XGBoost
        # model_input_df has numeric dtypes (what XGBoost expects)
        model_for_dice = SanitizedModel(predictor_model, model_input_df)

        dice_model = dice_ml.Model(
            model=model_for_dice,
            backend="sklearn",
            model_type=config.model_type,
        )

        # sklearn supports: random, genetic, kdtree
        return dice_ml.Dice(
            dice_data,
            dice_model,
            method=explainer_method,
        )

    # -----------------------------
    # TF2 BACKEND
    # -----------------------------
    elif config.backend == "TF2":
        dice_model = dice_ml.Model(
            model=predictor_model,
            backend="TF2",
            model_type=config.model_type,
        )

        # TF2 supports ONLY gradientdescent
        if explainer_method != "gradientdescent":
            raise ValueError(
                f"TF2 backend only supports 'gradientdescent', not '{explainer_method}'."
            )

        return DiceTensorFlow2(
            data_interface=dice_data,
            model_interface=dice_model,
        )

    # -----------------------------
    # Unsupported backend
    # -----------------------------
    else:
        raise ValueError(f"Unsupported backend for DiCE: {config.backend}")
