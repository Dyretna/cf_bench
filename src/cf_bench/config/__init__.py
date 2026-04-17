"""
Configuration module for counterfactual explanation pipelines.

This module defines configuration classes and explainer profiles used by the
DiCE-based counterfactual generation system. It provides a unified interface for
specifying model targets, feature metadata, and explainer-specific parameters.
"""

from .config import SystemConfig
from .paths import (  # train; test
    CF_OUTPUTS,
    DATA_DIR,
    MODEL_PATH_HB,
    MODEL_PATH_HC,
    MODELS_DIR,
    TEST_DATA_PATH_HB,
    TEST_DATA_PATH_HC,
    TRAIN_DATA_PATH_HB,
    TRAIN_DATA_PATH_HC,
)
from .profiles import (
    BaseExplainerProfile,
    GeneticExplainerProfile,
    GradientExplainerProfile,
    RandomExplainerProfile,
)

__all__ = [
    # configs
    "SystemConfig",
    # profiles
    "BaseExplainerProfile",
    "GeneticExplainerProfile",
    "GradientExplainerProfile",
    "RandomExplainerProfile",
    # paths
    "MODELS_DIR",
    "MODEL_PATH_HB",
    "MODEL_PATH_HC",
    "DATA_DIR",
    "TRAIN_DATA_PATH_HB",
    "TRAIN_DATA_PATH_HC",
    "TEST_DATA_PATH_HB",
    "TEST_DATA_PATH_HC",
    "CF_OUTPUTS",
]
