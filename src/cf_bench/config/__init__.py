"""
Configuration module for counterfactual explanation pipelines.

This module defines configuration classes and explainer profiles used by the
DiCE-based counterfactual generation system. It provides a unified interface for
specifying model targets, feature metadata, and explainer-specific parameters.
"""

from .config import SystemConfig
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
]
