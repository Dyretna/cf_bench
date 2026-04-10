import datetime as dt
import warnings
from pathlib import Path

import joblib
import pandas as pd

from .config import (  # GradientExplainerProfile, <-- not in use, no NN model
    CF_OUTPUTS,
    DATA_DIR,
    MODELS_DIR,
    TEST_DATA_PATH_HB,
    TEST_DATA_PATH_HC,
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

TEST_PATH = DATA_DIR / "07_from_HL" / "ete.csv"

# MODEL_PATH_FORMAT = MODELS_DIR / "rf_unbalanced" /"rf_{target}_2026-04-02.pkl"
MODEL_PATH = MODELS_DIR / "rf_HL_04_09.pkl"

# -----------------------------------------------------------------------------
#   Run each target and profile in main...
# -----------------------------------------------------------------------------


def main():
    targets = ["hltprhc"]

    explainer_profiles = [
        RandomExplainerProfile,
        # GeneticExplainerProfile,
    ]

    for target in targets:
        config = SystemConfig(target=target)

        for profile_cls in explainer_profiles:
            explainer_profile = profile_cls(features_to_vary=config.features_to_vary)
            run_explainer_for_target(config, explainer_profile)


# --- The Runner ---
# ==================


def run_explainer_for_target(config: SystemConfig, explainer_profile):
    target = config.target

    # --- load model ---
    predictor_model = joblib.load(str(MODEL_PATH))

    # --- load model ---
    df = pd.read_csv(TEST_PATH, index_col=0)
    df = df[config.feature_cols + [target]]

    # --- prepare query instances for DiCE ---
    top20_for_dice, original_index = prepare_query_instances(df, target, n=20)

    # --- create DiCE pipeline ---
    pipeline = DiceCFPipeline(
        config=config,
        explainer_profile=explainer_profile,
        predictor=predictor_model,
    )

    # --- run DiCE in the cleaned query instances ---
    (all_annotated, all_recommendations, all_formated_recs) = pipeline.process_batch(
        df, top20_for_dice
    )

    # --- build annotated batch ---
    annotated_batch = build_annotated_batch(top20_for_dice, all_annotated, target)

    # --- Restore original dataset index everywhere ---
    # Map DiCE query_index -> original dataset index
    index_map = dict(enumerate(original_index))

    annotated_batch["query_index"] = annotated_batch["query_index"].map(index_map)

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


def prepare_query_instances(df: pd.DataFrame, target: str, n: int = 20):
    """
    Select top-N positive instances for DiCE and return:
    - df_for_dice: a copy with a clean 0..N index
    - original_index: the original dataset index for each selected row
    """
    # query_df = df.loc[df[target] == 1]   # or df[target] if boolean

    # Select top-N rows WITHOUT filtering on target
    topn = df.head(n)

    # Save original dataset index
    original_index = topn.index.copy()

    # Remove target column before sending to DiCE
    df_for_dice = topn.drop(columns=[target]).reset_index(drop=True)

    return df_for_dice, original_index


if __name__ == "__main__":
    main()
