import datetime as dt
import warnings
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score

# local
from .config import (  # GradientExplainerProfile, <-- not in use, no NN model
    GeneticExplainerProfile,
    GradientExplainerProfile,
    RandomExplainerProfile,
    SystemConfig,
)
from .dice_pipeline import DiceRecommender, build_explainer, build_risk_evaluator
from .utils import annotate_all, build_annotated_batch, format_all, recommend_all

# DiCE random explainer triggers Pandas FutureWarnings due to
# internal .at[] assignments. (floats are wrong Dtype)
# Suppress them to keep output clean.
warnings.filterwarnings("ignore", category=FutureWarning)

PROFILE_MAP = {
    "random": RandomExplainerProfile,
    "genetic": GeneticExplainerProfile,
    "gradient": GradientExplainerProfile,
}


def run_pipeline(cfg):
    """Main runner: orchestrates data loading, CF generation, annotation and export.

    It wires together the existing components in a clear, explicit sequence.
    """
    # --- Configuration and model ---
    config = SystemConfig(target=cfg["target"], backend=cfg["backend"])

    if config.backend == "TF2":
        model = tf.keras.models.load_model(cfg["model_path"])
        is_keras = True
    else:
        model = joblib.load(cfg["model_path"])
        is_keras = False

    # --- Data loading ---
    df_raw = _load_data(cfg, config)

    # --- Optional scaling (for Keras models) ---
    df_for_model, scaler = _maybe_scale_data(cfg, config, df_raw, is_keras)

    # Query instances are taken from the same representation as the model sees
    query_df = df_for_model.head(cfg["n_query_instances"]).drop(columns=config.target)

    # --- Explainer profile selection ---
    profile_cls = PROFILE_MAP[cfg["explainer_profile"]]
    explainer_profile = profile_cls(features_to_vary=config.features_to_vary)

    # ---------------------------------------------------------------------------
    # Build DiCE explainer
    # and generate counterfactuals for the query instances.
    # ---------------------------------------------------------------------------

    explainer = build_explainer(
        config=config,
        predictor_model=model,
        df=df_for_model,
        explainer_method=explainer_profile.method,
    )
    cf_result = explainer.generate_counterfactuals(
        query_instances=query_df,
        **explainer_profile.to_cf_params(),
    )

    # --------------------------------------------------------------------------
    # Setup risk-evaluation for investigate validation of generated CF
    # recommendations are a list of dictionary of changed CF values.
    # formatted recommendations is a human readable friendly version.
    #
    # For exportation, all of above is collected in the 'all' funktions.
    # specially the annotate_all function is later used to to create a
    # bigger dataframe saved as a csv. used for comparisions between runs / models.
    # --------------------------------------------------------------------------

    risk_evaluator = build_risk_evaluator(
        backend=cfg["backend"],
        model=model,
        feature_cols=config.feature_cols,
        target_factor=config.target_factor,
    )

    recommender = DiceRecommender(
        feature_cols=config.feature_cols,
        target=config.target,
    )

    all_annotated = annotate_all(risk_evaluator, cf_result, query_df)
    all_recs = recommend_all(recommender, all_annotated, query_df)

    # ----------------------------------------------------------------------
    # Build annotated batch for export/inspection
    # Inverse scaling of structured data (back to raw space)
    #    - annotated_batch: for CSV / comparisons
    #    - recommendations: for human-readable recommendations
    # Note: we use the same query_df that was passed to DiCE/risk evaluator.
    # ----------------------------------------------------------------------

    annotated_batch = build_annotated_batch(
        query_instances=query_df,
        all_annotated=all_annotated,
        target=cfg["target"],
    )

    if scaler is not None:
        annotated_batch = _inverse_scale_batch(
            annotated_batch,
            scaler,
            config.feature_cols,
        )
        all_recs = _inverse_scale_recommendations(
            all_recs,
            scaler,
            config.feature_cols,
        )

    # Now we are in raw feature space for anything user-facing
    all_formatted = format_all(recommender, all_recs, query_df)

    # ---------------------------------------------------------------------------
    # Export of:
    #   - configs
    #   - predictor model
    #   - batch results
    # ---------------------------------------------------------------------------

    # --- Predictions for performance export ---
    # Use the same representation as the model was trained on (scaled or raw).
    y_true, y_pred = _compute_predictions(config, model, df_for_model)
    if config.backend == "TF2":
        y_pred = (y_pred >= 0.5).astype(int)

    report = classification_report(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    # --- Output directory and exports ---
    output_dir = _prepare_output_dir(cfg, explainer_profile)

    # Save annotated batch
    annotated_batch.to_csv(
        output_dir / f"{explainer_profile.method}_annotated_{cfg['target']}.csv",
        index=False,
    )

    suffix = f"{explainer_profile.method}_{config.target}.txt"

    # --- Save config ---
    with open(output_dir / f"config_{suffix}", "w", encoding="utf-8") as f:
        f.write("=== CONFIGURATION ===\n\n")
        f.write(str(config) + "\n\n")
        f.write(str(explainer_profile))

    # --- Save formatted recommendations (already formatted) ---
    rec_path = output_dir / f"recs_{suffix}"
    with open(rec_path, "w", encoding="utf-8") as f:
        f.write("=== RECOMMENDATIONS ===\n\n")
        for formatted in all_formatted:
            f.write(formatted)
            f.write("\n\n" + "=" * 80 + "\n\n")

    # --- Save model info and performance ---
    model_info_path = output_dir / f"model_{config.target}_info.txt"
    with open(model_info_path, "w", encoding="utf-8") as f:
        f.write("=== MODEL INFO ===\n\n")
        if config.backend == "sklearn":
            f.write("=== Sklearn model params ===\n")
            for k, v in model.get_params().items():
                f.write(f"{k}: {v}\n")
        else:
            f.write("Keras model summary:\n")
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        f.write("\n=== PERFORMANCE ===\n")
        f.write(report)
        f.write(f"\nROC-AUC: {roc_auc:.4f}\n")


# -------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------


def _load_data(cfg, config):
    """Load test data with correct dtypes and column ordering."""
    dtype_map = {
        **{col: int for col in config.ordinal_features},
        **{col: float for col in config.continuous_features},
    }

    df = pd.read_csv(cfg["test_path"], dtype=dtype_map)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df[config.feature_cols + [cfg["target"]]]
    return df


def _maybe_scale_data(cfg, config, df, is_keras):
    """Optionally scale data if the underlying model is a Keras model."""
    if not is_keras:
        return df, None

    scaler = joblib.load(cfg["scaler_path"])
    df_scaled = df.copy()
    df_scaled[config.feature_cols] = scaler.transform(df[config.feature_cols])
    return df_scaled, scaler


def _inverse_scale_batch(
    df: "pd.DataFrame",
    scaler,
    feature_cols: list[str],
) -> "pd.DataFrame":
    """Inverse-transform feature columns back to raw space."""
    df_copy = df.copy()

    # Clean up DiCE garbage values
    df_copy[feature_cols] = df_copy[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_copy[feature_cols] = scaler.inverse_transform(df_copy[feature_cols])
    return df_copy


def _inverse_scale_recommendations(
    all_recs,
    scaler,
    feature_cols: list[str],
):
    """Inverse-transform feature values inside recommendation dicts."""
    out = []
    for recs in all_recs:
        new_recs = []
        for r in recs:
            r2 = r.copy()
            # Only inverse-transform keys that are actual features
            feature_values = []
            feature_names = []
            for col in feature_cols:
                if col in r2:
                    feature_names.append(col)
                    feature_values.append(r2[col])

            if feature_values:
                inv_values = scaler.inverse_transform([feature_values])[0]
                for name, val in zip(feature_names, inv_values):
                    r2[name] = float(val)

            new_recs.append(r2)
        out.append(new_recs)
    return out


def _compute_predictions(config, model, df):
    """Compute model predictions for export and evaluation."""
    X = df[config.feature_cols]
    y_true = df[config.target]
    y_pred = model.predict(X)
    return y_true, y_pred


def _prepare_output_dir(cfg, explainer_profile):
    """Create a timestamped output directory for this run."""
    today = dt.datetime.today().strftime("%Y-%m-%d")
    run_name = f"{explainer_profile.method}_{cfg['target']}_{today}"
    output_dir = Path(cfg["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
