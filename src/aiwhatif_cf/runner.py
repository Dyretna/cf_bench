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

    df = pd.read_csv(
        cfg["test_path"], index_col=0 if cfg.get("use_original_index") else None
    )
    df = df[config.feature_cols + [cfg["target"]]]

    if cfg.get("use_original_index"):
        topn = df.head(cfg["n_query_instances"])
        original_index = topn.index.copy()
        query_df = topn.drop(columns=[cfg["target"]]).reset_index(drop=True)
    else:
        q = df.loc[df[cfg["target"]] == 1, config.feature_cols]
        query_df = q.head(cfg["n_query_instances"])
        original_index = None

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

    if original_index is not None:
        index_map = dict(enumerate(original_index))
        annotated_batch["query_index"] = annotated_batch["query_index"].map(index_map)

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
