# =============================================================================
# dice_batch_runner.py — Main pipeline orchestrator
# =============================================================================
#
# PURPOSE:
#   This file is the "conductor" of the entire project.
#   It runs the full counterfactual generation pipeline for a batch of people
#   (one person at a time in a loop), then saves the results to CSV.
#
# HOW TO USE:
#   Called via the CLI:
#     python -m cf_bench.cli --config configs/rf_hltprhc_cfcheck.yaml
#   The YAML config file specifies which model, data, and algorithm to use.
#
# PIPELINE STEPS (in order):
#   1. Load model (Random Forest, XGBoost, or Neural Network)
#   2. Load training data (DiCE uses it to understand realistic feature values)
#   3. Load test data (the people we want to generate CFs for)
#   4. Format data for DiCE (type conversions, scaling if needed)
#   5. Build the DiCE explainer with the chosen search algorithm
#   6. Loop over each person → generate counterfactuals → record time
#   7. Evaluate the risk before/after each CF
#   8. Assemble results into a table
#   9. Compute model accuracy and timing metrics
#  10. Save everything to cf_outputs/
#
# OUTPUT:
#   A folder in cf_outputs/ containing:
#     - random_annotated_hltprhc.csv  → the main results table
#     - config_*.txt                  → settings used for this run
#     - model_*_info.json/.txt        → model details and accuracy
# =============================================================================


import logging
import time
import warnings

import joblib
import pandas as pd

# Local imports
from .config import (
    BaseExplainerProfile,
    GeneticExplainerProfile,
    GradientExplainerProfile,
    RandomExplainerProfile,
    SystemConfig,
)
from .data.loaders import load_dice_compatible_data
from .data.transformers import FeatureScaler, QueryInstancePreparer
from .dice_adapters import SanitizedModel, build_explainer, build_risk_evaluator
from .results.exporters import (
    ConfigExporter,
    ModelInfoExporter,
    create_output_directory,
)
from .results.metrics import PerformanceMetrics
from .results.model_info_extractors import extract_model_info
from .results.predictions import ModelPredictor
from .utils import annotate_all, build_annotated_batch

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# DiCE random explainer triggers Pandas FutureWarnings due to
# internal .at[] assignments. (floats are wrong Dtype)
# Suppress them to keep output clean.
# --------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

# The three search strategies DiCE can use to find counterfactuals.
# All three have the same goal: find a modified version of the person
# where the model's prediction changes. They differ in HOW they search:
#   random   → randomly tries combinations of feature values (fast, simple)
#   genetic  → evolves solutions over many generations, like natural selection (smarter, slower)
#   gradient → uses the model's math to navigate toward a flip (only works for neural nets)
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
    # SystemConfig is a "rulebook" for this experiment: which features exist,
    # which are ordinal (survey categories), which can be changed by DiCE, etc.
    config = SystemConfig(target=cfg["target"], backend=cfg["backend"])

    # Load the trained model from disk.
    # joblib is used for sklearn models (Random Forest, XGBoost).
    # Keras (TF2) is used for neural networks and needs a different loader.
    if config.backend == "TF2":
        import tensorflow as tf

        model = tf.keras.models.load_model(cfg["model_path"])
        is_keras = True
    else:
        model = joblib.load(cfg["model_path"])
        is_keras = False

    # --- Data loading ---
    # We need TWO datasets:
    #   df_training_for_dice → DiCE uses this to learn what feature values are realistic
    #                          (e.g. BMI ranges from 18-40 in training data → won't suggest BMI=200)
    #   df_raw               → the actual people we want to generate CFs for (test set)
    df_traning_for_dice = load_dice_compatible_data(cfg["train_path"], config)
    df_raw = load_dice_compatible_data(cfg["test_path"], config)

    # --- Optional scaling (for Keras/neural network models only) ---
    # Neural networks require all features to be on the same small scale (0-1).
    # Example: BMI=28 and alcfreq=3 are very different magnitudes — scaling fixes this.
    # Random Forest and XGBoost do NOT need scaling, so this block is skipped for them.
    if is_keras:
        scaler = joblib.load(cfg["scaler_path"])
        feature_scaler = FeatureScaler(scaler, config)
        model_input_df = feature_scaler.transform(df_raw, convert_ordinals=True)
    else:
        model_input_df = df_raw
        scaler = None

    # --- Query instances preparation for DiCE ---
    # A "query instance" = one specific person we want a counterfactual for.
    # DiCE is very strict about data types — this step formats each person's
    # data to match exactly what DiCE expects (e.g. ordinal values as strings).
    preparer = QueryInstancePreparer(config)
    query_df = preparer.prepare(model_input_df, df_traning_for_dice)

    # --- Prepare numeric reference for SanitizedModel ---
    # TYPE MISMATCH PROBLEM:
    #   DiCE treats ordinal survey features (etfruit, cgtsmok, etc.) as STRINGS
    #   internally, because it handles them like categories ("3", "4", "6"...).
    #   But the Random Forest model expects NUMBERS (3, 4, 6...).
    #
    # SOLUTION: keep two versions of the data:
    #   model_input_df         → strings  → for DiCE
    #   model_input_df_numeric → numbers  → for the RF/XGBoost model
    #
    # SanitizedModel is a thin wrapper around the real model.
    # When DiCE passes it string data, SanitizedModel converts to numbers first.
    model_input_df_numeric = model_input_df.copy()
    for col in config.ordinal_features:
        if col in model_input_df_numeric.columns:
            model_input_df_numeric[col] = pd.to_numeric(model_input_df_numeric[col])

    logger.debug("model_input_df dtypes (for SanitizedModel):")
    for col in config.feature_cols:
        if col in model_input_df_numeric.columns:
            logger.debug(f"  {col}: {model_input_df_numeric[col].dtype}")

    logger.debug("DiCE dtype consistency check:")
    logger.debug(
        f"Training data dtype (etfruit): {df_traning_for_dice['etfruit'].dtype}"
    )
    logger.debug(f"Query data dtype (etfruit): {query_df['etfruit'].dtype}")
    logger.debug(
        "Training unique values (etfruit): "
        f"{sorted(df_traning_for_dice['etfruit'].unique())}"
    )
    logger.debug(
        f"Query unique values (etfruit): {sorted(query_df['etfruit'].unique())}"
    )

    # --- Explainer profile selection ---
    # Pick the search algorithm from the config (random / genetic / gradient).
    # The profile also carries all DiCE parameters:
    #   features_to_vary   → which features DiCE is allowed to change
    #   total_CFs          → how many counterfactuals to generate per person
    #   stopping_threshold → stop early if CF risk drops below this fraction of original risk
    #   use_permitted_range → whether to enforce realistic value bounds per feature
    profile_cls = PROFILE_MAP[cfg["explainer_profile"]]

    explainer_profile: BaseExplainerProfile = profile_cls(
        features_to_vary=config.features_to_vary,
        ordinal_allowed_values=config.ordinal_allowed_values,
        total_CFs=cfg["total_CFs"],
        stopping_threshold=cfg["stopping_threshold"],
        use_permitted_range=cfg.get("use_permitted_range", True),  # Default to True
    )

    # --------------------------------------------------------------------------
    # Build DiCE explainer
    # --------------------------------------------------------------------------
    # build_explainer creates the DiCE object that will generate counterfactuals.
    # DiCE needs to know: the model, the training data (for feature bounds),
    # and which search method to use (random / genetic / gradient).
    # The numeric version of the data is passed here because DiCE uses it
    # only as a dtype reference — the actual search uses string ordinals.
    explainer = build_explainer(
        config=config,
        predictor_model=model,
        model_input_df=model_input_df_numeric,  # Use numeric version for dtype reference
        training_df=df_traning_for_dice,
        explainer_method=explainer_profile.method,
    )

    # --------------------------------------------------------------------------
    # Main loop: generate counterfactuals for each person in the test set
    # --------------------------------------------------------------------------
    # We process one person at a time (not all at once) because:
    #   - DiCE needs per-row "permitted ranges" (directional bounds per person)
    #   - We time each person individually for performance analysis
    #
    # For each person:
    #   single_query = that person's features formatted as a 1-row dataframe
    #   cf           = DiCE's result: a list of modified versions of the person
    #                  where the model predicts a better health outcome
    logger.info(f"Generating counterfactuals for {len(query_df)} instances...")
    cf_results = []
    cf_times = []  # time taken per person, becomes cf_gen_time_sec in the CSV

    for idx, row in query_df.iterrows():
        # Convert this single row to a 1-row dataframe (DiCE requires a dataframe, not a Series)
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
    # Risk evaluation
    # ---------------------------------------------------------------------------
    # DiCE returns counterfactuals but does NOT compute the actual risk percentage.
    # We pass each CF back through the model to get:
    #   risk_before          → original person's risk (e.g. 6.3%)
    #   target_risk          → goal (set to half of risk_before by default)
    #   predicted_risk_after → risk after applying the CF (e.g. 2.0%)
    #   valid                → True if predicted_risk_after is below target_risk
    #
    # SanitizedModel wraps the real model and handles the string→number conversion
    # needed because DiCE outputs string values for ordinal features.
    if cfg["backend"] == "sklearn":
        model_for_risk = SanitizedModel(model, model_input_df_numeric)
    else:
        model_for_risk = model

    risk_evaluator = build_risk_evaluator(
        backend=cfg["backend"],
        model=model_for_risk,
        feature_cols=config.feature_cols,
        target_factor=config.target_factor,
    )

    # annotate_all → adds risk columns to every CF result
    all_annotated = annotate_all(risk_evaluator, cf_results, query_df)

    # ----------------------------------------------------------------------
    # Assemble the final output table
    # ----------------------------------------------------------------------
    # build_annotated_batch creates the CSV structure you see in cf_outputs/:
    #   - one "original" row per person (their actual feature values + risk)
    #   - one "cf_1", "cf_2", ... row per counterfactual (only changed features shown)
    # Empty cells in CF rows mean that feature was NOT changed by this CF.
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

    # Inverse scale if needed (neural network models only)
    # If we scaled the data earlier (step 3), the CF feature values are still
    # in 0-1 scale. We convert them back to real units (e.g. BMI=0.4 → BMI=24)
    # so the output CSV is human-readable.
    if scaler is not None:
        feature_scaler = FeatureScaler(scaler, config)
        annotated_batch = feature_scaler.inverse_transform(
            annotated_batch,
            feature_cols=config.feature_cols,
        )

    # -------------------------------------------------------------------------
    # Model accuracy and timing metrics
    # -------------------------------------------------------------------------
    # Compute two sets of metrics to save alongside the CFs:
    #   performance_metrics → how accurate is the model? (accuracy, F1, etc.)
    #                         y_true = real labels from test data
    #                         y_pred = model's predictions on test data
    #   timing_metrics      → how long did CF generation take?
    #                         (average, min, max across all persons)
    predictor = ModelPredictor(backend=config.backend)
    y_true, y_pred = predictor.predict(
        model=model,
        df=model_input_df_numeric,
        feature_cols=config.feature_cols,
        target_col=config.target,
    )
    performance_metrics = PerformanceMetrics.compute_classification_metrics(
        y_true, y_pred
    )
    timing_metrics = PerformanceMetrics.compute_timing_metrics(cf_times)

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    logger.info("Exporting results...")

    # Extract and model info
    model_info = extract_model_info(model, config)

    output_dir = create_output_directory(
        output_base=cfg["output_dir"],
        model_type=model_info["model_type"],
        explainer_method=explainer_profile.method,
        target=cfg["target"],
    )
    logger.info(f"Output directory: {output_dir}")

    # Save annotated batch CSV
    annotated_batch.to_csv(
        output_dir / f"{explainer_profile.method}_annotated_{cfg['target']}.csv",
        index=False,
    )

    # Export configuration with timing
    suffix = f"{explainer_profile.method}_{config.target}.txt"
    ConfigExporter.export(
        output_path=output_dir / f"config_{suffix}",
        system_config=config,
        explainer_profile=explainer_profile,
        timing_metrics=timing_metrics,
    )

    # export modell info
    ModelInfoExporter.export_json(
        output_dir / f"model_{config.target}_info.json",
        model_info,
    )

    ModelInfoExporter.export(
        output_dir / f"model_{config.target}_info.txt",
        model_info,
        performance_metrics,
    )
