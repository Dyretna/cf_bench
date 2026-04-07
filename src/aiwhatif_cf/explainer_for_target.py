import datetime as dt
import warnings
from pathlib import Path

import joblib
import pandas as pd

from .config import (  # GradientExplainerProfile, <-- not in use, no NN model
    CF_OUTPUTS,
    MODELS_DIR,
    TEST_DATA_PATH_HB,
    TEST_DATA_PATH_HC,
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


# -----------------------------------------------------------------------------
#   Set test paths to dict - to be used dynamically on loop
# -----------------------------------------------------------------------------

TEST_PATHS = {
    "hltprhb": TEST_DATA_PATH_HB,
    "hltprhc": TEST_DATA_PATH_HC,
}

MODEL_PATH_FORMAT = MODELS_DIR / "rf_{target}_2026-04-07.pkl"

# -----------------------------------------------------------------------------
#   Run each target and profile in main...
# -----------------------------------------------------------------------------


def main():
    targets = ["hltprhb", "hltprhc"]

    explainer_profiles = [
        RandomExplainerProfile,
        GeneticExplainerProfile,
    ]

    for target in targets:
        config = SystemConfig(target=target)

        for profile_cls in explainer_profiles:
            explainer_profile = profile_cls(features_to_vary=config.features_to_vary)
            run_explainer_for_target(config, explainer_profile)


# --- The Runner ---
# ==================


def run_explainer_for_target(config, explainer_profile):
    target = config.target

    # load model and data
    predictor_model = joblib.load(str(MODEL_PATH_FORMAT).format(target=target))

    df = pd.read_csv(TEST_PATHS[target])
    df = df[config.feature_cols + [target]]

    # create pipeline
    pipeline = DiceCFPipeline(
        config=config,
        explainer_profile=explainer_profile,
        predictor=predictor_model,
    )

    # select query instances for pipeline
    # later transform from json (dict) to df
    query_instances_df = df.loc[df[target] == 1, config.feature_cols]
    top5 = query_instances_df.head(5)

    (all_annotated, all_recommendations, all_formated_recs) = pipeline.process_batch(
        df, top5
    )

    annotated_batch = build_annotated_batch(top5, all_annotated, target)

    print("\n=== Annoted Batch ===")
    print(annotated_batch)

    # Compute model predictions for performance export
    X = df[config.feature_cols]
    y_true = df[target]
    y_pred = predictor_model.predict(X)

    # --- Export everything  ---
    today = dt.datetime.today().strftime("%Y-%m-%d")
    run_name = f"{explainer_profile.method}_{target}_{today}"
    output_dir = Path(CF_OUTPUTS) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save annotated batch
    annotated_batch.to_csv(
        output_dir / f"{explainer_profile.method}_annotated_{target}.csv", index=False
    )

    export_batch_results(
        output_dir=output_dir,
        formatted_recommendations=all_formated_recs,
        config=config,
        explainer_profile=explainer_profile,
        rf_model=predictor_model,
        y_true=y_true,
        y_pred=y_pred,
    )


if __name__ == "__main__":
    main()
