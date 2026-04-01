from dataclasses import dataclass, field
from typing import Optional


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
    random_seed: int = 42

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

    population_size: int = 50  # example values
    mutation_rate: float = 0.1  # example values
    crossover_rate: float = 0.5  # example values
    diversity_weight: float = 1.0  # example values
    sample_size: int = 1000  # example values

    permitted_range: Optional[dict] = None
    features_to_vary: Optional[list[str]] = None
    random_seed: int = 42

    def to_cf_params(self):
        return {
            "total_CFs": self.total_CFs,
            "desired_class": self.desired_class,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "diversity_weight": self.diversity_weight,
            "sample_size": self.sample_size,
            "permitted_range": self.permitted_range,
            "features_to_vary": self.features_to_vary,
            "random_seed": self.random_seed,
        }

    def __str__(self):
        header = "=== Genetic Explainer Profile ==="
        lines = [f"{key:23}: {value}" for key, value in self.__dict__.items()]
        return header + "\n" + "\n".join(lines)


# Gradient explainer --> good for NN-models
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
