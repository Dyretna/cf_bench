import datetime as dt
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# local
from .config import (
    BaseExplainerProfile,
    GeneticExplainerProfile,
    GradientExplainerProfile,
    RandomExplainerProfile,
    SystemConfig,
)
from .dice_adapters import DiceRecommender, build_explainer, build_risk_evaluator
from .dice_adapters.build_explainer import SanitizedModel
from .model_info_extractors import extract_model_info
from .utils import annotate_all, build_annotated_batch, format_all, recommend_all

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# DiCE random explainer triggers Pandas FutureWarnings due to
# internal .at[] assignments. (floats are wrong Dtype)
# Suppress them to keep output clean.
# --------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

PROFILE_MAP = {
    "random": RandomExplainerProfile,
    "genetic": GeneticExplainerProfile,
    "gradient": GradientExplainerProfile,
}

# --------------------------------------------------------------------------
# Run the batch pipeline
# --------------------------------------------------------------------------


def run_pipeline(cfg):
    """Main runner: orchestrates data loading, CF generation, annotation and export."""

    logger.info(f"Starting counterfactual pipeline for target: {cfg['target']}")
    logger.info(f"Explainer profile: {cfg['explainer_profile']}")

    # --- Configuration and model ---
    config = SystemConfig(target=cfg["target"], backend=cfg["backend"])

    if config.backend == "TF2":
        import tensorflow as tf

        model = tf.keras.models.load_model(cfg["model_path"])
        is_keras = True
    else:
        model = joblib.load(cfg["model_path"])
        is_keras = False

    # --- Data loading ---
    df_traning_for_dice = _load_data(cfg["train_path"], config)
    df_raw = _load_data(cfg["test_path"], config)

    # --- Optional scaling (for Keras models) ---
    model_input_df, scaler = _maybe_scale_data(cfg, config, df_raw, is_keras)

    # --------------------------------------------------------------------------
    # Query instances preparation for DiCE
    # --------------------------------------------------------------------------
    # Query instances are derived from model_input_df (which has numeric dtypes
    # for XGBoost compatibility). However, DiCE requires string dtypes for
    # ordinal/categorical features to perform internal validation.
    #
    # We therefore convert ordinal features to strings here, matching the
    # dtype of df_traning_for_dice (which was loaded with string dtypes for
    # ordinals via _load_data).
    # --------------------------------------------------------------------------
    query_df = model_input_df.drop(columns=config.target)

    # Convert ordinal features from numeric (XGBoost output) to string (DiCE input)
    for feat in config.ordinal_features:
        if feat in query_df.columns:
            # Round float64 values to nearest int, then convert to string
            # Only convert if currently numeric (avoid converting strings)
            if pd.api.types.is_numeric_dtype(query_df[feat]):
                query_df[feat] = query_df[feat].round().astype(int).astype(str)

    # Ensure query_df dtypes match training_df dtypes exactly
    query_df = query_df.astype(
        df_traning_for_dice.drop(columns=[config.target]).dtypes.to_dict()
    )

    # Ensure query_df dtypes match training_df dtypes exactly
    query_df = query_df.astype(
        df_traning_for_dice.drop(columns=[config.target]).dtypes.to_dict()
    )

    # Ensure model_input_df has NUMERIC dtypes for SanitizedModel
    # (it should already, but defensive programming)
    for col in config.ordinal_features:
        if col in model_input_df.columns:
            if model_input_df[col].dtype == "object":
                model_input_df[col] = pd.to_numeric(model_input_df[col])

    logger.debug("model_input_df dtypes (for SanitizedModel):")
    for col in config.feature_cols:
        if col in model_input_df.columns:
            logger.debug(f"  {col}: {model_input_df[col].dtype}")

    logger.debug("DiCE dtype consistency check:")
    logger.debug(
        f"Training data dtype (etfruit): {df_traning_for_dice['etfruit'].dtype}"
    )
    logger.debug(f"Query data dtype (etfruit): {query_df['etfruit'].dtype}")
    logger.debug(
        f"Training unique values (etfruit): {sorted(df_traning_for_dice['etfruit'].unique())}"
    )
    logger.debug(
        f"Query unique values (etfruit): {sorted(query_df['etfruit'].unique())}"
    )

    # --- Explainer profile selection ---
    profile_cls = PROFILE_MAP[cfg["explainer_profile"]]

    explainer_profile: BaseExplainerProfile = profile_cls(
        features_to_vary=config.features_to_vary,
        ordinal_allowed_values=config.ordinal_allowed_values,
        total_CFs=cfg["total_CFs"],
        stopping_threshold=cfg["stopping_threshold"],
        use_permitted_range=cfg.get("use_permitted_range", True),  # Default to True
    )

    # --------------------------------------------------------------------------
    # Build DiCE explainer and generate counterfactuals
    # --------------------------------------------------------------------------
    explainer = build_explainer(
        config=config,
        predictor_model=model,
        model_input_df=model_input_df,
        training_df=df_traning_for_dice,
        explainer_method=explainer_profile.method,
    )

    # --------------------------------------------------------------------------
    # Generate counterfactuals for each query instance
    # --------------------------------------------------------------------------
    # Note: query_df now has string dtypes for ordinals, matching training_df.
    # The explainer profile will also generate permitted_range with string
    # values for ordinals, ensuring consistency throughout the DiCE pipeline.
    # --------------------------------------------------------------------------
    logger.info(f"Generating counterfactuals for {len(query_df)} instances...")
    cf_results = []
    cf_times = []  # Track individual CF generation times

    for idx, row in query_df.iterrows():
        single_query = row.to_frame().T

        # Debug: show current query values and training data coverage
        logger.debug(f"Query instance {idx}:")
        logger.debug("Query values:")
        for feat in config.ordinal_features:
            if feat in row.index:
                logger.debug(
                    f"  {feat}: {row[feat]} (dtype: {type(row[feat]).__name__})"
                )

        logger.debug("Training coverage:")
        for feat in config.ordinal_features:
            if feat in df_traning_for_dice.columns:
                unique_vals = sorted(df_traning_for_dice[feat].unique())
                logger.debug(f"  {feat}: {unique_vals}")

        # Time CF generation for this instance
        start_time = time.perf_counter()
        cf = explainer.generate_counterfactuals(
            query_instances=single_query,
            **explainer_profile.to_cf_params_for_row(row),
        )
        end_time = time.perf_counter()

        cf_results.append(cf)
        cf_times.append(end_time - start_time)

    # ---------------------------------------------------------------------------
    # Risk evaluation and recommendation generation
    # --------------------------------------------------------------------------
    # The risk evaluator receives DiCE outputs (with string ordinals) and
    # SanitizedModel converts them back to numeric for XGBoost prediction.
    # --------------------------------------------------------------------------
    # Wrap model with SanitizedModel for sklearn backend to handle string→numeric conversion
    model_for_risk = (
        SanitizedModel(model, model_input_df) if cfg["backend"] == "sklearn" else model
    )

    risk_evaluator = build_risk_evaluator(
        backend=cfg["backend"],
        model=model_for_risk,
        feature_cols=config.feature_cols,
        target_factor=config.target_factor,
    )

    recommender = DiceRecommender(
        feature_cols=config.feature_cols,
        target=config.target,
    )

    all_annotated = annotate_all(risk_evaluator, cf_results, query_df)
    all_recs = recommend_all(recommender, all_annotated, query_df)

    # ----------------------------------------------------------------------
    # Build annotated batch for export/inspection
    # Inverse scaling of structured data (back to raw space)
    # ----------------------------------------------------------------------
    annotated_batch = build_annotated_batch(
        query_instances=query_df,
        all_annotated=all_annotated,
        target=cfg["target"],
    )

    # Add CF generation time per instance to annotated batch
    # Only set timing for 'original' rows (one time per query instance)
    # Round to 2 decimals for readability
    time_mapping = {idx: round(time, 2) for idx, time in zip(query_df.index, cf_times)}
    annotated_batch["cf_gen_time_sec"] = annotated_batch.apply(
        lambda row: time_mapping.get(row["query_index"])
        if row["cf_id"] == "original"
        else None,
        axis=1,
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
    all_formatted = format_all(
        recommender, all_recs, query_df, model_input_df[config.target]
    )

    # --- Predictions for performance export ---
    y_true, y_pred = _compute_predictions(config, model, model_input_df)
    if config.backend == "TF2":
        y_pred = (y_pred >= 0.5).astype(int)

    report = classification_report(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    # --------------------------------------------------------------------------
    # Export results
    # --------------------------------------------------------------------------
    logger.info("Exporting results...")
    output_dir = _prepare_output_dir(cfg, explainer_profile)
    logger.info(f"Output directory: {output_dir}")

    # Save annotated batch
    annotated_batch.to_csv(
        output_dir / f"{explainer_profile.method}_annotated_{cfg['target']}.csv",
        index=False,
    )

    suffix = f"{explainer_profile.method}_{config.target}.txt"

    # --- Save config with timing info ---
    total_cf_time = sum(cf_times)
    avg_cf_time = total_cf_time / len(cf_times) if cf_times else 0
    min_cf_time = min(cf_times) if cf_times else 0
    max_cf_time = max(cf_times) if cf_times else 0

    with open(output_dir / f"config_{suffix}", "w", encoding="utf-8") as f:
        f.write("=== CONFIGURATION ===\n\n")
        f.write(str(config) + "\n\n")
        f.write(str(explainer_profile))
        f.write("\n\n=== TIMING INFORMATION ===\n\n")
        f.write(f"Total CF generation time: {total_cf_time:.2f}s\n")
        f.write(f"Average time per instance: {avg_cf_time:.2f}s\n")
        f.write(f"Min time: {min_cf_time:.2f}s\n")
        f.write(f"Max time: {max_cf_time:.2f}s\n")
        f.write(f"Number of instances: {len(cf_times)}\n")

    # --- Save formatted recommendations ---
    rec_path = output_dir / f"recs_{suffix}"
    with open(rec_path, "w", encoding="utf-8") as f:
        f.write("=== RECOMMENDATIONS ===\n\n")
        for formatted in all_formatted:
            f.write(formatted)
            f.write("\n\n" + "=" * 80 + "\n\n")

    # --- Save model info and performance ---
    # Extract model information dynamically
    model_info = extract_model_info(model, config)

    # Save as JSON for machine readability
    model_info_json_path = output_dir / f"model_{config.target}_info.json"
    with open(model_info_json_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, default=str)

    # Save human-readable version
    model_info_path = output_dir / f"model_{config.target}_info.txt"
    with open(model_info_path, "w", encoding="utf-8") as f:
        f.write("=== MODEL INFO ===\n\n")
        f.write(f"Model Type: {model_info['model_type']}\n")
        f.write(f"Model Class: {model_info['model_class']}\n")
        f.write(f"Model Module: {model_info['model_module']}\n\n")

        # Handle Keras models with summary
        if "summary" in model_info:
            f.write("Model Summary:\n")
            f.write(model_info["summary"])
            f.write(f"\nNumber of Layers: {model_info.get('num_layers', 'N/A')}\n")
            f.write(f"Total Parameters: {model_info.get('total_params', 'N/A')}\n")
        else:
            # For non-Keras models, write parameters
            f.write("Parameters:\n")
            params = model_info.get("params", {})
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")

        # Add feature importances if available
        if "feature_importances" in model_info:
            f.write("\nFeature Importances:\n")
            # Sort by importance (descending)
            sorted_features = sorted(
                model_info["feature_importances"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for feat, importance in sorted_features:
                f.write(f"  {feat}: {importance:.4f}\n")

        f.write("\n=== PERFORMANCE ===\n")
        f.write(report)
        f.write(f"\nROC-AUC: {roc_auc:.4f}\n")


# -------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------


def _load_data(path, config):
    """
    Load data with correct dtypes for DiCE.

    DiCE treats ordinal/categorical features as strings internally and performs
    strict equality checks. To ensure compatibility:
    - Ordinal features are loaded as strings
    - Continuous features are loaded as floats

    This ensures that DiCE's internal validation passes and that permitted_range
    values match training data dtypes exactly.

    Parameters
    ----------
    path : str
        Path to CSV file
    config : SystemConfig
        Configuration object containing feature metadata

    Returns
    -------
    pd.DataFrame
        DataFrame with correct dtypes and column ordering
    """
    dtype_map = {
        **{col: str for col in config.ordinal_features},  # strings for DiCE
        **{col: float for col in config.continuous_features},
    }

    df = pd.read_csv(path, dtype=dtype_map)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df[config.feature_cols + [config.target]]
    return df


def _maybe_scale_data(cfg, config, df, is_keras):
    """
    Optionally scale data if the underlying model is a Keras model.

    Note: Scaling is applied to the numeric representation of features.
    For ordinal features loaded as strings, we first convert to numeric,
    scale, then convert back to strings for DiCE compatibility.
    """
    if not is_keras:
        return df, None

    scaler = joblib.load(cfg["scaler_path"])
    df_scaled = df.copy()

    # Convert string ordinals to numeric for scaling
    for col in config.ordinal_features:
        if col in df_scaled.columns:
            df_scaled[col] = pd.to_numeric(df_scaled[col])

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
    """
    Compute model predictions for export and evaluation.

    Note: For XGBoost models, we need numeric dtypes. If df has string
    ordinals (from DiCE), we convert them to numeric first.
    """
    X = df[config.feature_cols].copy()

    # Convert string ordinals to numeric for model prediction
    for col in config.ordinal_features:
        if col in X.columns and X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col])

    y_true = df[config.target]

    # Handle string targets if present
    if y_true.dtype == "object":
        y_true = pd.to_numeric(y_true)

    y_pred = model.predict(X)
    return y_true, y_pred


def _prepare_output_dir(cfg, explainer_profile):
    """Create a timestamped output directory for this run."""
    today = dt.datetime.today().strftime("%Y-%m-%d")
    run_name = f"{explainer_profile.method}_{cfg['target']}_{today}"
    output_dir = Path(cfg["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
