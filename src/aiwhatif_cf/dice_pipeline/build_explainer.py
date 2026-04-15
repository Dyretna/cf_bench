import dice_ml
import pandas as pd

from ..config import SystemConfig


class SanitizedModel:
    """
    A generic model wrapper that enforces consistent input dtypes.

    DiCE generates synthetic counterfactual samples that may contain mixed or
    incorrect dtypes (e.g., 'object' instead of numeric). Some models such as
    XGBoost require strict numeric dtypes and will fail otherwise.

    This wrapper:
    - stores the dtypes from the training DataFrame
    - sanitizes all incoming DataFrames to match those dtypes
    - works for any sklearn-compatible model without special-casing
    """

    def __init__(self, model, train_df):
        self.model = model
        self.train_dtypes = train_df.dtypes.to_dict()

    def _sanitize(self, df) -> pd.DataFrame:
        """Casts columns in df to the stored training dtypes."""
        df = df.copy()
        for col, dtype in self.train_dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        return df

    def predict_proba(self, df):
        """Sanitizes df and forwards the call to the underlying model."""
        df = self._sanitize(df)
        return self.model.predict_proba(df)

    def predict(self, df):
        """Sanitizes df and forwards the call to the underlying model."""
        df = self._sanitize(df)
        return self.model.predict(df)


# ----------------------------------------------------------------------
#  Build DiCE explainer for both sklearn and TF2 backends
# ----------------------------------------------------------------------


def build_explainer(config: SystemConfig, predictor_model, df, explainer_method):
    """
    Build and return a DiCE explainer instance.

    Parameters
    ----------
    config : DiceConfig
        Configuration object containing feature metadata, backend, model type,
        and default explainer method.
    predictor_model : Any
        The trained predictive model to wrap inside the DiCE Model object.
    df : pandas.DataFrame
        The dataset used to construct the DiCE Data object.
    explainer_method : str,
        Defines the DiCE explainer type,
        eg. 'random', 'genetic' or 'gradientdescent'

    Returns
    -------
    dice_ml.Dice
        A fully constructed DiCE explainer ready to generate counterfactuals.
    """

    dice_data = dice_ml.Data(
        dataframe=df,
        continuous_features=config.continuous_features,
        outcome_name=config.target,
    )

    # Decide how to present the model to DiCE based on backend
    if config.backend == "sklearn":
        # Wrap sklearn-like models to sanitize dtypes
        model_for_dice = SanitizedModel(predictor_model, df)
    elif config.backend == "TF2":
        # For Keras/TF2 models, pass the raw model (callable)
        model_for_dice = predictor_model

    else:
        raise ValueError(f"Unsupported backend for DiCE: {config.backend}")

    dice_model = dice_ml.Model(
        model=model_for_dice,
        backend=config.backend,  # sklearn
        model_type=config.model_type,  # classifer
    )

    return dice_ml.Dice(
        dice_data,
        dice_model,
        method=explainer_method,
    )
