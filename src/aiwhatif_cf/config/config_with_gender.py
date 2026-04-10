"""
Configuration module for counterfactual explanation pipelines.

This module defines configuration classes and explainer profiles used by the
DiCE-based counterfactual generation system. It provides a unified interface for
specifying model targets, feature metadata, and explainer-specific parameters.

------------------------------------------------------------------------------
SystemConfig
------------------------------------------------------------------------------
SystemConfig represents the configuration for a *single* model target. It
contains:

- target (str): The outcome variable to explain.
- feature_cols (list[str]): All model input features.
- immutable_cols (list[str]): Features that must not be changed in CFs.
- continuous_features (list[str]): Features treated as continuous.
- features_to_vary (list[str]): Automatically computed as all features except
  immutable ones.
- backend, model_type, target_factor: Additional metadata used by the pipeline.

A SystemConfig instance corresponds to one full counterfactual run for one
target variable.

------------------------------------------------------------------------------
Explainer Profiles
------------------------------------------------------------------------------
Each explainer profile defines the parameters passed to DiCE when generating
counterfactuals. The profiles expose only the parameters that DiCE actually
supports for each explainer type.

### RandomExplainerProfile
Used for tree-based or tabular models when a simple baseline method is desired.

Supported parameters:
- total_CFs
- desired_class
- features_to_vary
- permitted_range
- sample_size
- random_seed

Random explainer does *not* use proximity, sparsity, or diversity weights.

### GeneticExplainerProfile
Used for models where DiCE's internal genetic algorithm is appropriate.

Important:
DiCE's GeneticExplainer does **not** expose GA hyperparameters such as:
- population_size
- mutation_rate
- crossover_rate
- diversity_weight
- sample_size

These parameters exist in some academic GA implementations but are *not*
supported by DiCE. The only valid parameters are:

Supported parameters:
- total_CFs
- desired_class
- features_to_vary
- permitted_range
- random_seed
- posthoc_sparsity_param (optional)
- maxiterations (optional)
- verbose (optional)

### GradientExplainerProfile
Used for differentiable models (e.g., neural networks).

Supported parameters:
- total_CFs
- desired_class
- proximity_weight
- sparsity_weight
- diversity_weight
- categorical_penalty
- features_to_vary
- permitted_range
- random_seed

Gradient explainer supports optimization-based tuning, unlike the genetic
explainer.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
A typical pipeline run constructs:

    config = SystemConfig(target="hltprhb")
    explainer = GeneticExplainerProfile(features_to_vary=config.features_to_vary)
    pipeline = DiceCFPipeline(config=config, explainer_profile=explainer, ...)

The explainer profile's `to_cf_params()` method returns only the parameters
supported by the corresponding DiCE explainer.

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- DiCE explainers differ significantly in which parameters they accept.
- Passing unsupported parameters will raise TypeError at runtime.
- This module ensures that only valid parameters are forwarded to DiCE.
"""

from dataclasses import dataclass, field
from typing import Optional

# ------------------------------------------------------------------------------
#   System Config
# ------------------------------------------------------------------------------


@dataclass
class SystemConfig:
    target: str

    feature_cols: list[str] = field(
        default_factory=lambda: [
            "etfruit",
            "eatveg",
            "cgtsmok",
            "alcfreq",
            "slprl",
            "paccnois",
            "bmi",
            "gndr",
            "dosprt",
        ]
    )
    immutable_cols: list[str] = field(default_factory=lambda: ["gndr"])
    continuous_features: list[str] = field(default_factory=lambda: ["bmi"])

    backend: str = "sklearn"
    model_type: str = "classifier"
    target_factor: float = 0.5  # multiplier for RiskEvaluator

    def __post_init__(self):
        self.features_to_vary = [
            c for c in self.feature_cols if c not in self.immutable_cols
        ]

    def __str__(self):
        header = "=== System Config ==="
        lines = [f"{key:23}: {value}" for key, value in self.__dict__.items()]
        return header + "\n" + "\n".join(lines)


# ------------------------------------------------------------------------------
#   Explainer Profiles
# ------------------------------------------------------------------------------


@dataclass
class RandomExplainerProfile:
    method: str = "random"
    total_CFs: int = 3
    desired_class: int = 0
    stopping_threshold: float = 0.5
    posthoc_sparsity_param: float = 0.1
    posthoc_sparsity_algorithm: str = "linear"
    permitted_range: Optional[dict] = None
    features_to_vary: Optional[list[str]] = None
    random_seed: int = 101

    def to_cf_params(self):
        return {
            "total_CFs": self.total_CFs,
            "desired_class": self.desired_class,
            "permitted_range": self.permitted_range,
            "features_to_vary": self.features_to_vary,
            "stopping_threshold": self.stopping_threshold,
            "posthoc_sparsity_param": self.posthoc_sparsity_param,
            "posthoc_sparsity_algorithm": self.posthoc_sparsity_algorithm,
            "random_seed": self.random_seed,
        }

    def __str__(self):
        header = "=== Random Explainer Profile ==="
        lines = [f"{key:23}: {value}" for key, value in self.__dict__.items()]
        return header + "\n" + "\n".join(lines)


# Genetic explainer --> good for tree models
@dataclass
class GeneticExplainerProfile:
    method: str = "genetic"
    total_CFs: int = 3
    desired_class: int = 0
    features_to_vary: Optional[list[str]] = None
    permitted_range: Optional[dict] = None
    stopping_threshold: float = 0.5
    posthoc_sparsity_param: float = 0.1
    posthoc_sparsity_algorithm: str = "linear"
    maxiterations: int = 500
    verbose = False

    def to_cf_params(self):
        return {
            "total_CFs": self.total_CFs,
            "desired_class": self.desired_class,
            "features_to_vary": self.features_to_vary,
            "permitted_range": self.permitted_range,
            "stopping_threshold": self.stopping_threshold,
            "posthoc_sparsity_param": self.posthoc_sparsity_param,
            "posthoc_sparsity_algorithm": self.posthoc_sparsity_algorithm,
            "maxiterations": self.maxiterations,
            "verbose": self.verbose,
        }

    def __str__(self):
        header = "=== Genetic Explainer Profile ==="
        lines = [f"{key:23}: {value}" for key, value in self.__dict__.items()]
        return header + "\n" + "\n".join(lines)


# ----- NOT IN USE FOR RF MODELS ----


# Gradient explainer --> for NN-models
@dataclass
class GradientExplainerProfile:
    method: str = "gradientdescent"
    total_CFs: int = 3
    desired_class: int = 0

    proximity_weight: float = 0.5  # example values
    sparsity_weight: float = 0.5  # example values
    diversity_weight: float = 1.0  # example values
    categorical_penalty: float = 0.1  # example values

    permitted_range: Optional[dict] = None
    features_to_vary: Optional[list[str]] = None
    random_seed: int = 42

    def to_cf_params(self):
        return {
            "total_CFs": self.total_CFs,
            "desired_class": self.desired_class,
            "proximity_weight": self.proximity_weight,
            "sparsity_weight": self.sparsity_weight,
            "diversity_weight": self.diversity_weight,
            "categorical_penalty": self.categorical_penalty,
            "permitted_range": self.permitted_range,
            "features_to_vary": self.features_to_vary,
            "random_seed": self.random_seed,
        }

    def __str__(self):
        header = "=== GD Explainer Profile ==="
        lines = [f"{key:23}: {value}" for key, value in self.__dict__.items()]
        return header + "\n" + "\n".join(lines)
