"""
Reporter package for counterfactual analysis.

Simplified functional approach with dataclasses (as per Results_Summary_Analysis_Plan.md).
"""

from .config_parser import (
    ConfigParser,
    ExperimentConfig,
    extract_ml_model_type,
    find_config_file,
)
from .summary import ExperimentSummary, generate_comparison_report, summarize_experiment

__all__ = [
    # Config parsing
    "ConfigParser",
    "ExperimentConfig",
    "find_config_file",
    "extract_ml_model_type",
    # Summary generation
    "ExperimentSummary",
    "summarize_experiment",
    "generate_comparison_report",
]
