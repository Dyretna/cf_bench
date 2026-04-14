import datetime as dt
import warnings
from pathlib import Path

import joblib
import pandas as pd

from .config import (  # GradientExplainerProfile, <-- not in use, no NN model
    GeneticExplainerProfile,
    RandomExplainerProfile,
    SystemConfig,
)
from .dice_pipeline import DiceCFPipeline
from .utils import build_annotated_batch, export_batch_results

# DiCE random explainer triggers Pandas FutureWarnings due to
# internal .at[] assignments. (floats are wrong Dtype)
# Suppress them to keep output clean.
warnings.filterwarnings("ignore", category=FutureWarning)

PROFILE_MAP = {
    "random": RandomExplainerProfile,
    "genetic": GeneticExplainerProfile,
}


def run_pipeline(cfg):
    config = SystemConfig(target=cfg["target"])
    model = joblib.load(cfg["model_path"])

    dtype_map = {
        **{col: int for col in config.ordinal_features},
        **{col: float for col in config.continuous_features},
    }

    df = pd.read_csv(cfg["test_path"], dtype=dtype_map)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df[config.feature_cols + [cfg["target"]]]
    query_df = df.head(cfg["n_query_instances"]).drop(columns=config.target)
    print(query_df.info())

    profile_cls = PROFILE_MAP[cfg["explainer_profile"]]
    explainer_profile = profile_cls(features_to_vary=config.features_to_vary)

    # create pipeline
    pipeline = DiceCFPipeline(
        config=config,
        explainer_profile=explainer_profile,
        predictor=model,
    )

    (all_annotated, all_recommendations, all_formated_recs) = pipeline.process_batch(
        df, query_df
    )

    annotated_batch = build_annotated_batch(query_df, all_annotated, cfg["target"])

    print("\n=== Annoted Batch ===")
    print(annotated_batch)

    # Compute model predictions for performance export
    X = df[config.feature_cols]
    y_true = df[cfg["target"]]
    y_pred = model.predict(X)

    # --- Export everything  ---
    today = dt.datetime.today().strftime("%Y-%m-%d")
    run_name = f"{explainer_profile.method}_{cfg['target']}_{today}"
    output_dir = Path(cfg["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save annotated batch
    annotated_batch.to_csv(
        output_dir / f"{explainer_profile.method}_annotated_{cfg['target']}.csv",
        index=False,
    )

    export_batch_results(
        output_dir=output_dir,
        formatted_recommendations=all_formated_recs,
        config=config,
        explainer_profile=explainer_profile,
        rf_model=model,
        y_true=y_true,
        y_pred=y_pred,
    )
