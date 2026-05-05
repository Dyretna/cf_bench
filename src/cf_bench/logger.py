import logging

logger = logging.getLogger(__name__)


class LoggingHelper:
    """Helper class for structured debug logging throughout the pipeline."""

    @staticmethod
    def log_model_input_dtypes(model_input_df_numeric, feature_cols):
        logger.debug("model_input_df dtypes (for SanitizedModel):")
        for col in feature_cols:
            if col in model_input_df_numeric.columns:
                logger.debug(f"  {col}: {model_input_df_numeric[col].dtype}")

    @staticmethod
    def log_dtype_consistency(df_traning_for_dice, query_df):
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

    @staticmethod
    def log_query_and_training_coverage(idx, row, config, df_traning_for_dice):
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
