import dice_ml

from ..config import SystemConfig


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

    dice_model = dice_ml.Model(
        model=predictor_model,
        backend=config.backend,
        model_type=config.model_type,
    )

    return dice_ml.Dice(
        dice_data,
        dice_model,
        method=explainer_method,
    )
