import dice_ml
import pandas as pd

from ..config import SystemConfig


class SanitizedModel:
    def __init__(self, model, train_df):
        self.model = model
        self.train_dtypes = train_df.dtypes.to_dict()

    def _sanitize(self, df):
        df = df.copy()
        for col, dtype in self.train_dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        return df

    def predict_proba(self, df):
        df = self._sanitize(df)
        return self.model.predict_proba(df)

    def predict(self, df):
        df = self._sanitize(df)
        return self.model.predict(df)


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

    sanitized = SanitizedModel(predictor_model, df)

    dice_model = dice_ml.Model(
        model=sanitized,
        backend=config.backend,  # sklearn
        model_type=config.model_type,  # classifer
    )

    return dice_ml.Dice(
        dice_data,
        dice_model,
        method=explainer_method,
    )
