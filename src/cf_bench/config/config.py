"""
------------------------------------------------------------------------------
SystemConfig
------------------------------------------------------------------------------
SystemConfig represents the configuration for a *single* model target.
It is the central place where we define how a trained model should be used
for counterfactual generation - without needing access to the original
training data.

Each SystemConfig instance corresponds to one full counterfactual run for
one target variable.

Core responsibilities
---------------------
- Describe which features the model uses.
- Describe which features are continuous vs. ordinal.
- Describe which feature values are allowed for ordinal features.
- Provide metadata needed by the explainer backend (e.g. DiCE).

Fields
------
- target (str):
    The outcome variable to explain (e.g. a risk flag or probability
    thresholded into a class).

- backend (str):
    Which explainer backend is used (e.g. "sklearn", "TF2", "pytorch").
    This is used by the pipeline to choose the correct explainer and
    model wrapper.

- model_type (str):
    The type of model, typically "classifier". Included for clarity and
    potential future branching.

- feature_cols (list[str]):
    All model input features, in the exact order expected by the model.
    This is the canonical feature schema for the model.

- ordinal_features (list[str]):
    Subset of feature_cols that are treated as ordinal / discrete
    (e.g. Likert scales, frequency categories). These features do not
    vary continuously, but only over a fixed set of allowed values.

- continuous_features (list[str]):
    Subset of feature_cols that are treated as continuous (e.g. BMI).
    These features can vary over an interval rather than a fixed set
    of discrete categories.

- ordinal_allowed_values (dict[str, list[int]]):
    For each ordinal feature, this dictionary defines the *complete*
    set of allowed values that the model was trained on.

    This is crucial for counterfactual generation, especially for
    backends like sklearn/DiCE that require explicit permitted ranges
    for categorical/ordinal features.

    Instead of deriving allowed values from training data at runtime,
    we store them here as part of the model's schema. That means:

        - We do not need to ship training data into production.
        - We avoid inconsistencies from using test data or ad-hoc
          unique() calls.
        - Counterfactuals are guaranteed to stay within the model's
          valid input domain.

    Example:
        "etfruit": [1, 2, 3, 4, 5, 6, 7]
        "dosprt":  [0, 1, 2, 3, 4, 5, 6, 7]

    The values below come from an offline analysis of the
    training data (e.g. unique values per feature) and are then
    hard-coded here as part of the model configuration.

    The allowed values will be sent to each DiCE profile, as it is
    responsible of building the explainer.

- features_to_vary (list[str]):
    Computed in __post_init__ as all features in feature_cols
    (optionally excluding immutable features if those are added).
    This list is passed to the explainer to indicate which features
    are allowed to change in counterfactuals.

- target_factor (float):
    A scaling factor used by downstream components such as a
    RiskEvaluator. It can be used to adjust thresholds or risk
    sensitivity for a given target.

"""

from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    backend: str
    model_type: str = "classifier"
    target: str = "hltprhc"

    feature_cols: list[str] = field(
        default_factory=lambda: [
            "etfruit",
            "eatveg",
            "cgtsmok",
            "alcfreq",
            "slprl",
            "paccnois",
            "bmi",
            "dosprt",
        ]
    )
    # immutable_cols: list[str] = field(default_factory=lambda: ["gndr"])
    ordinal_features: list[str] = field(
        default_factory=lambda: [
            "etfruit",
            "eatveg",
            "cgtsmok",
            "alcfreq",
            "slprl",
            "paccnois",
            "dosprt",
        ]
    )
    continuous_features: list[str] = field(default_factory=lambda: ["bmi"])

    ordinal_allowed_values: dict[str, list[int]] = field(
        default_factory=lambda: {
            "etfruit": [1, 2, 3, 4, 5, 6, 7],
            "eatveg": [1, 2, 3, 4, 5, 6, 7],
            "cgtsmok": [1, 2, 3, 4, 5, 6],
            "alcfreq": [1, 2, 3, 4, 5, 6, 7],
            "slprl": [1, 2, 3, 4],
            "paccnois": [0, 1],
            "dosprt": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )

    target_factor: float = 0.5  # multiplier for RiskEvaluator

    def __post_init__(self):
        self.features_to_vary = [
            c
            for c in self.feature_cols  # if c not in self.immutable_cols
        ]

    def __str__(self):
        header = "=== System Config ==="
        lines = [f"{key:23}: {value}" for key, value in self.__dict__.items()]
        return header + "\n" + "\n".join(lines)
