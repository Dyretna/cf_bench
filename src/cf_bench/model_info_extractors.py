"""Dynamic model information extraction using registry pattern.

This module provides a flexible system for extracting model information
from different types of machine learning models (XGBoost, RandomForest,
Keras, sklearn-compatible models).
"""

from typing import Any


class ModelInfoExtractor:
    """Base class for extracting model information."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        """Return True if this extractor can handle this model."""
        raise NotImplementedError

    @staticmethod
    def extract(model: Any, config: Any) -> dict:
        """Extract model information as a dictionary."""
        raise NotImplementedError


class XGBoostExtractor(ModelInfoExtractor):
    """Extractor for XGBoost models."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        model_name = type(model).__name__
        return model_name in ["XGBClassifier", "XGBRegressor"]

    @staticmethod
    def extract(model: Any, config: Any) -> dict:
        info = {
            "model_type": "XGBoost",
            "model_class": type(model).__name__,
            "model_module": type(model).__module__,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "params": model.get_params(),
        }

        # Add feature importances if available
        if hasattr(model, "feature_importances_"):
            info["feature_importances"] = dict(
                zip(config.feature_cols, model.feature_importances_.tolist())
            )

        return info


class RandomForestExtractor(ModelInfoExtractor):
    """Extractor for RandomForest models."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        model_name = type(model).__name__
        return model_name in ["RandomForestClassifier", "RandomForestRegressor"]

    @staticmethod
    def extract(model: Any, config: Any) -> dict:
        info = {
            "model_type": "RandomForest",
            "model_class": type(model).__name__,
            "model_module": type(model).__module__,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "params": model.get_params(),
        }

        # Add feature importances if available
        if hasattr(model, "feature_importances_"):
            info["feature_importances"] = dict(
                zip(config.feature_cols, model.feature_importances_.tolist())
            )

        return info


class KerasExtractor(ModelInfoExtractor):
    """Extractor for Keras/TensorFlow models."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        # Duck typing for Keras models
        return hasattr(model, "summary") and hasattr(model, "layers")

    @staticmethod
    def extract(model: Any, config: Any) -> dict:
        from io import StringIO

        stream = StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))

        info = {
            "model_type": "Keras/TensorFlow",
            "model_class": type(model).__name__,
            "model_module": type(model).__module__,
            "summary": stream.getvalue(),
            "num_layers": len(model.layers),
            "total_params": int(model.count_params()),
        }

        return info


class SklearnExtractor(ModelInfoExtractor):
    """Fallback extractor for any sklearn-compatible model."""

    @staticmethod
    def can_handle(model: Any) -> bool:
        return hasattr(model, "get_params")

    @staticmethod
    def extract(model: Any, config: Any) -> dict:
        info = {
            "model_type": "sklearn-compatible",
            "model_class": type(model).__name__,
            "model_module": type(model).__module__,
            "params": model.get_params(),
        }

        # Add feature importances if available
        if hasattr(model, "feature_importances_"):
            info["feature_importances"] = dict(
                zip(config.feature_cols, model.feature_importances_.tolist())
            )

        return info


# Registry - order matters, first match wins
# Keep SklearnExtractor last as it's the most generic fallback
EXTRACTORS = [
    XGBoostExtractor,
    RandomForestExtractor,
    KerasExtractor,
    SklearnExtractor,  # Fallback (last in list)
]


def extract_model_info(model: Any, config: Any) -> dict:
    """Extract model information using the appropriate extractor.

    Args:
        model: The trained model instance
        config: Configuration object containing feature columns and other metadata

    Returns:
        Dictionary containing model information

    """
    for extractor_cls in EXTRACTORS:
        if extractor_cls.can_handle(model):
            return extractor_cls.extract(model, config)

    # Ultimate fallback if no extractor can handle the model
    return {
        "model_type": "unknown",
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
    }
