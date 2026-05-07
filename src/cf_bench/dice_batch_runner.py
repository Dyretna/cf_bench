# =============================================================================
# dice_batch_runner.py — Main pipeline orchestrator
# =============================================================================
#
# PURPOSE:
#   This file is the "conductor" of the entire project.
#   It runs the full counterfactual generation pipeline for a batch of people
#   (one person at a time in a loop), then saves the results to CSV.
#
# WHAT IS A BATCH?
#   Instead of running DiCE for just one person, we run it for all people
#   in the test dataset (cfcheck.csv). "Batch" = many people at once.
#
# HOW TO USE:
#   Called via the CLI:
#     python -m cf_bench.cli --config configs/xgboost_genetic_sparsity_01.yaml
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
#   9. Inverse Scale the data (if needed, for Neural Network)
#  10. Compute model accuracy and timing metrics
#  11. Save everything to cf_outputs/
#
# OUTPUT:
#   A folder in cf_outputs/ containing:
#     - annotated.csv       → the main results table
#     - config_*.txt        → settings used for this run
#     - model_info.json/txt → model details and accuracy
# =============================================================================


import logging
import time
import warnings

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
from .logger import LoggingHelper
from .model_loader import load_model_by_backend
from .results.assembler import assemble_annotated_results
from .results.exporters import (
    ConfigExporter,
    ModelInfoExporter,
    create_output_directory,
)
from .results.metrics import PerformanceMetrics
from .results.model_info_extractors import extract_model_info
from .results.predictions import predict_model

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


class BatchRunner:
    """Orchestrates counterfactual generation pipeline."""

    def __init__(self, cfg):
        # Store config values as attributes
        self.train_path = cfg["train_path"]
        self.test_path = cfg["test_path"]
        self.model_path = cfg["model_path"]
        self.scaler_path = cfg.get("scaler_path")
        self.output_dir = cfg["output_dir"]
        self.backend = cfg["backend"]
        self.explainer_profile_name = cfg["explainer_profile"]
        self.total_CFs = cfg["total_CFs"]
        self.stopping_threshold = cfg["stopping_threshold"]
        self.use_permitted_range = cfg.get("use_permitted_range", True)

        # --- Feature locking ---
        # Features listed in features_to_lock will NOT be changed by DiCE.
        # Example YAML: features_to_lock: ["cgtsmok", "bmi"]
        # Use case: person who cannot quit smoking → lock cgtsmok → DiCE finds other changes.
        self.features_to_lock = cfg.get("features_to_lock", [])

        # --- Sparsity parameter ---
        # Controls post-processing: higher = fewer features changed per CF.
        # 0.1 = minimal trimming (many features can change)
        # 0.9 = aggressive trimming (as few features as possible change)
        self.posthoc_sparsity_param = cfg.get("posthoc_sparsity_param", 0.1)

        # --- Run identifier ---
        # Optional label appended to the output folder name to distinguish runs.
        # Example YAML: run_id: "sp0.5"  → folder: XGBoost_genetic_prange_lowthres_2026-04-30_sp0.5
        self.run_id = cfg.get("run_id")

        # System config is the "rulebook": which features exist, which are ordinal,
        # which can be changed by DiCE, etc.
        self.config = SystemConfig(backend=self.backend)

        # Build the filtered features_to_vary list (all features except locked ones)
        features_to_vary = [
            f for f in self.config.feature_cols if f not in self.features_to_lock
        ]

        if self.features_to_lock:
            logger.info(
                f"Locked features (DiCE will not change): {self.features_to_lock}"
            )
            logger.info(f"Features DiCE can vary: {features_to_vary}")

        # Load model from disk (no training happens here — model was pre-trained)
        self.model, self.is_keras, self.scaler = load_model_by_backend(
            backend=self.backend,
            model_path=self.model_path,
            scaler_path=self.scaler_path,
        )

        # Create explainer profile
        # (DiCE Search Algorithm: "random" | "genetic" | "gradient", + params)
        profile_cls = PROFILE_MAP[self.explainer_profile_name]
        self.explainer_profile: BaseExplainerProfile = profile_cls(
            features_to_vary=features_to_vary,
            ordinal_allowed_values=self.config.ordinal_allowed_values,
            total_CFs=self.total_CFs,
            stopping_threshold=self.stopping_threshold,
            use_permitted_range=self.use_permitted_range,
            posthoc_sparsity_param=self.posthoc_sparsity_param,
        )

    # ==========================================================================
    # The Runner
    # ==========================================================================

    def run(self):
        """Main runner: orchestrates data loading, CF generation, annotation and export."""

        logger.info("Starting counterfactual pipeline")
        logger.info(f"Predictor model: {type(self.model).__name__}")
        logger.info(f"Explainer profile: {self.explainer_profile_name}")
        logger.info(f"stopping threshold: {self.stopping_threshold}")
        logger.info(f"Sparsity param: {self.posthoc_sparsity_param}")
        if self.features_to_lock:
            logger.info(f"Locked features: {self.features_to_lock}")

        config = self.config

        # --- Data loading ---
        # We need TWO datasets:
        #   df_training_for_dice → DiCE uses this to learn what feature values are realistic
        #   df_raw               → the actual people we want to generate CFs for (test set)
        df_traning_for_dice = load_dice_compatible_data(self.train_path, config)
        df_raw = load_dice_compatible_data(self.test_path, config)

        # --- Optional scaling (for Keras/neural network models only) ---
        # Neural networks require all features on the same small scale (0-1).
        # Random Forest and XGBoost do NOT need scaling.
        if self.is_keras:
            feature_scaler = FeatureScaler(self.scaler, config)
            model_input_df = feature_scaler.transform(df_raw, convert_ordinals=True)
        else:
            model_input_df = df_raw

        # --- Query instances preparation for DiCE ---
        # A "query instance" = one specific person we want a counterfactual for.
        # DiCE is very strict about data types — this step formats each person's
        # data to match exactly what DiCE expects (e.g. ordinal values as strings).
        preparer = QueryInstancePreparer(self.config)
        query_df = preparer.prepare(model_input_df, df_traning_for_dice)

        # --- Prepare numeric reference for SanitizedModel ---
        # TYPE MISMATCH PROBLEM:
        #   DiCE treats ordinal features as STRINGS ("3", "4"...).
        #   But Random Forest / XGBoost expect NUMBERS (3, 4...).
        # SOLUTION: keep two versions:
        #   model_input_df         → strings → for DiCE
        #   model_input_df_numeric → numbers → for the model via SanitizedModel
        model_input_df_numeric = model_input_df.copy()
        for col in config.ordinal_features:
            if col in model_input_df_numeric.columns:
                model_input_df_numeric[col] = pd.to_numeric(model_input_df_numeric[col])

        LoggingHelper.log_model_input_dtypes(
            model_input_df_numeric, config.feature_cols
        )
        LoggingHelper.log_dtype_consistency(df_traning_for_dice, query_df)

        # --------------------------------------------------------------------------
        # Build DiCE explainer
        # --------------------------------------------------------------------------
        # build_explainer creates the DiCE object that will generate counterfactuals.
        # DiCE needs: the model, the training data (for feature bounds),
        # and which search method to use (random / genetic / gradient).
        explainer = build_explainer(
            config=config,
            predictor_model=self.model,
            model_input_df=model_input_df_numeric,
            training_df=df_traning_for_dice,
            explainer_method=self.explainer_profile.method,
        )

        # --------------------------------------------------------------------------
        # Main loop: generate counterfactuals for each person in the test set
        # --------------------------------------------------------------------------
        # We process one person at a time because DiCE needs per-row permitted ranges.
        # For each person:
        #   single_query = that person's features as a 1-row dataframe
        #   cf           = DiCE's result: modified versions where prediction improves
        logger.info(f"Generating counterfactuals for {len(query_df)} instances...")
        cf_results = []
        cf_times = []  # time taken per person, becomes cf_gen_time_sec in the CSV

        for idx, row in query_df.iterrows():
            single_query = row.to_frame().T
            LoggingHelper.log_query_and_training_coverage(
                idx, row, config, df_traning_for_dice
            )
            # Time CF generation for this instance
            start_time = time.perf_counter()
            cf = explainer.generate_counterfactuals(
                query_instances=single_query,
                **self.explainer_profile.to_cf_params_for_row(row),
            )
            end_time = time.perf_counter()
            cf_results.append(cf)
            cf_times.append(end_time - start_time)
            # --- gower distance here?? ---

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
        # SanitizedModel wraps the real model and handles the string→number conversion.
        if self.backend == "sklearn":
            model_for_risk = SanitizedModel(self.model, model_input_df_numeric)
        else:
            model_for_risk = self.model

        risk_evaluator = build_risk_evaluator(
            backend=self.backend,
            model=model_for_risk,
            feature_cols=config.feature_cols,
            target_factor=config.target_factor,
        )

        # annotate_all → adds risk_before, predicted_risk_after, and valid to each CF
        all_annotated = risk_evaluator.annotate_all(cf_results, query_df)

        # ----------------------------------------------------------------------
        # Assemble the final output table with all metrics
        # ----------------------------------------------------------------------
        # Builds the CSV structure with:
        #   - one "original" row per person (their actual feature values + risk)
        #   - one "cf_1", "cf_2", ... row per counterfactual (only changed features shown)
        #   - timing data, Gower distance, nchanged count, and proper column ordering
        annotated_batch = assemble_annotated_results(
            query_df=query_df,
            all_annotated=all_annotated,
            cf_times=cf_times,
            target=config.target,
            feature_cols=config.feature_cols,
        )

        # Inverse scale if needed (neural network models only)
        # Convert scaled values back to real units (e.g. BMI=0.4 → BMI=24)
        if self.scaler is not None:
            feature_scaler = FeatureScaler(self.scaler, config)
            annotated_batch = feature_scaler.inverse_transform(
                annotated_batch,
                feature_cols=config.feature_cols,
            )

        # -------------------------------------------------------------------------
        # Model performance and timing metrics
        # -------------------------------------------------------------------------

        # Evaluate model accuracy on the test set (accuracy, F1, ROC-AUC)
        y_true, y_pred = predict_model(
            backend=config.backend,
            model=self.model,
            df=model_input_df_numeric,
            feature_cols=config.feature_cols,
            target_col=config.target,
        )
        performance_metrics = PerformanceMetrics.compute_classification_metrics(
            y_true, y_pred
        )

        # Compute CF generation timing statistics (mean, median, min, max)
        timing_metrics = PerformanceMetrics.compute_timing_metrics(cf_times)

        # -------------------------------------------------------------------------
        # Export
        # -------------------------------------------------------------------------
        logger.info("Exporting results...")

        model_info = extract_model_info(self.model, config)

        output_dir = create_output_directory(
            output_base=self.output_dir,
            model_type=model_info["model_type"],
            threshold=self.explainer_profile.stopping_threshold,
            run_id=self.run_id,
        )
        logger.info(f"Output directory: {output_dir}")

        # Save annotated batch CSV
        annotated_batch.to_csv(output_dir / "annotated.csv", index=False)

        # Export configuration with timing
        suffix = f"{model_info['model_type']}_{self.explainer_profile.method}.txt"
        ConfigExporter.export(
            output_path=output_dir / f"config_{suffix}",
            system_config=config,
            explainer_profile=self.explainer_profile,
            timing_metrics=timing_metrics,
        )

        ModelInfoExporter.export_json(
            output_dir / "model_info.json",
            model_info,
        )

        ModelInfoExporter.export(
            output_dir / "model_info.txt",
            model_info,
            performance_metrics,
        )


# --------------------------------------------------------------------------
# Entry point called by cli.py
# --------------------------------------------------------------------------


def run_pipeline(cfg):
    """Run the counterfactual pipeline with the given configuration."""
    runner = BatchRunner(cfg)
    runner.run()
