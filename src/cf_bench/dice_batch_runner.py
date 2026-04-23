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

        # System config
        self.config = SystemConfig(backend=self.backend)

        # Load model and scaler once
        self.model, self.is_keras, self.scaler = self._load_model()

        # Create explainer profile
        # (Dice Search Algorithm: "random" | "genetic" | "gradient", + params)
        profile_cls = PROFILE_MAP[self.explainer_profile_name]
        self.explainer_profile: BaseExplainerProfile = profile_cls(
            features_to_vary=self.config.features_to_vary,
            ordinal_allowed_values=self.config.ordinal_allowed_values,
            total_CFs=self.total_CFs,
            stopping_threshold=self.stopping_threshold,
            use_permitted_range=self.use_permitted_range,
        )

    def run(self):
        """Main runner: orchestrates data loading, CF generation, annotation and export."""

        logger.info("Starting counterfactual pipeline")
        logger.info(f"Predictor model: {type(self.model).__name__}")
        logger.info(f"Explainer profile: {self.explainer_profile_name}")
        logger.info(f"stopping threshold: {self.stopping_threshold}")

        config = self.config

        # --- Data loading ---
        df_traning_for_dice = load_dice_compatible_data(self.train_path, config)
        df_raw = load_dice_compatible_data(self.test_path, config)

        # --- Optional scaling (for Keras models) ---
        if self.is_keras:
            feature_scaler = FeatureScaler(self.scaler, config)
            model_input_df = feature_scaler.transform(df_raw, convert_ordinals=True)
        else:
            model_input_df = df_raw

        # --- Query instances preparation for DiCE ---
        # Create Query preparer
        preparer = QueryInstancePreparer(self.config)
        query_df = preparer.prepare(model_input_df, df_traning_for_dice)

        # --- Prepare numeric reference for SanitizedModel ---
        # SanitizedModel needs to know what dtypes the MODEL expects (numeric for ordinals).
        # model_input_df has strings, so we create a numeric version as reference.
        model_input_df_numeric = model_input_df.copy()
        for col in config.ordinal_features:
            if col in model_input_df_numeric.columns:
                model_input_df_numeric[col] = pd.to_numeric(model_input_df_numeric[col])

        self._log_model_input_dtypes(model_input_df_numeric, config.feature_cols)
        self._log_dtype_consistency(df_traning_for_dice, query_df)

        # --------------------------------------------------------------------------
        # Build DiCE explainer and generate counterfactuals
        # --------------------------------------------------------------------------
        explainer = build_explainer(
            config=config,
            predictor_model=self.model,
            model_input_df=model_input_df_numeric,  # Use numeric version for dtype reference
            training_df=df_traning_for_dice,
            explainer_method=self.explainer_profile.method,
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
            self._log_query_and_training_coverage(idx, row, config, df_traning_for_dice)
            # Time CF generation for this instance
            start_time = time.perf_counter()
            cf = explainer.generate_counterfactuals(
                query_instances=single_query,
                # purpose is to set direction and constraints
                # in permitted_range for each query(row)
                **self.explainer_profile.to_cf_params_for_row(row),
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
        # Wrap model with SanitizedModel for sklearn backend
        # to handle string -> numeric conversion
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

        all_annotated = annotate_all(risk_evaluator, cf_results, query_df)

        # ----------------------------------------------------------------------
        # Build annotated batch for export/inspection
        # Inverse scaling of structured data (back to raw space)
        # ----------------------------------------------------------------------
        annotated_batch = build_annotated_batch(
            query_instances=query_df,
            all_annotated=all_annotated,
            target=config.target,
        )

        # Add CF generation time per instance to annotated batch
        # Only set timing for 'original' rows (one time per query instance)
        # Round to 2 decimals for readability
        time_mapping = {
            idx: round(time, 2) for idx, time in zip(query_df.index, cf_times)
        }
        annotated_batch["cf_gen_time_sec"] = annotated_batch.apply(
            lambda row: time_mapping.get(row["query_index"])
            if row["cf_id"] == "original"
            else None,
            axis=1,
        )

        # Inverse scale if needed (if TF2 backend)
        if self.scaler is not None:
            feature_scaler = FeatureScaler(self.scaler, config)
            annotated_batch = feature_scaler.inverse_transform(
                annotated_batch,
                feature_cols=config.feature_cols,
            )

        # -------------------------------------------------------------------------
        # Prediction and metrics
        # -------------------------------------------------------------------------
        predictor = ModelPredictor(backend=config.backend)
        y_true, y_pred = predictor.predict(
            model=self.model,
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
        model_info = extract_model_info(self.model, config)

        output_dir = create_output_directory(
            output_base=self.output_dir,
            model_type=model_info["model_type"],
            explainer_method=self.explainer_profile.method,
            threshold=self.explainer_profile.stopping_threshold,
            use_permitted_range=self.use_permitted_range,
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

        # export modell info
        ModelInfoExporter.export_json(
            output_dir / "model_info.json",
            model_info,
        )

        ModelInfoExporter.export(
            output_dir / "model_info.txt",
            model_info,
            performance_metrics,
        )

    # -------------------------------------------------------------------------
    # Setup helpers
    # -------------------------------------------------------------------------

    def _load_model(self):
        """Load model and scaler. Returns (model, is_keras, scaler)."""
        if self.config.backend == "TF2":
            import tensorflow as tf

            model = tf.keras.models.load_model(self.model_path)
            is_keras = True
            scaler = joblib.load(self.scaler_path)
        else:
            model = joblib.load(self.model_path)
            is_keras = False
            scaler = None
        return model, is_keras, scaler

    # -------------------------------------------------------------------------
    # Logger debug helpers
    # -------------------------------------------------------------------------

    def _log_model_input_dtypes(self, model_input_df_numeric, feature_cols):
        logger.debug("model_input_df dtypes (for SanitizedModel):")
        for col in feature_cols:
            if col in model_input_df_numeric.columns:
                logger.debug(f"  {col}: {model_input_df_numeric[col].dtype}")

    def _log_dtype_consistency(self, df_traning_for_dice, query_df):
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

    def _log_query_and_training_coverage(self, idx, row, config, df_traning_for_dice):
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


# --------------------------------------------------------------------------
# Run the batch pipeline
# --------------------------------------------------------------------------


def run_pipeline(cfg):
    """Run the counterfactual pipeline with the given configuration."""
    runner = BatchRunner(cfg)
    runner.run()
