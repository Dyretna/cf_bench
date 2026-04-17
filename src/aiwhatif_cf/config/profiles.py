"""
-----------------------------------------------------------------------------
Explainer Profiles (with ABC)
-----------------------------------------------------------------------------
This module defines a clean abstraction for DiCE explainer profiles.

- BaseExplainerProfile (ABC):
    Handles:
        * permitted_range merging
        * directional bounds
        * to_cf_params()
        * to_cf_params_for_row()
    Subclasses only define:
        * method name
        * which DiCE parameters they support

### RandomExplainerProfile
Used for tree-based or tabular models when a simple baseline method is desired.
Random explainer does *not* use proximity, sparsity, or diversity weights.

### GeneticExplainerProfile
Used for models where DiCE's internal genetic algorithm is appropriate.

### GradientExplainerProfile
Used for differentiable models (e.g., neural networks).
Gradient explainer supports optimization-based tuning, unlike the genetic
explainer.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
A typical pipeline run constructs:

    config = SystemConfig(target="hltprhb")
    explainer = GeneticExplainerProfile(features_to_vary=config.features_to_vary)
    cf = explainer.generate_counterfactuals(
        query_instances=single_query,
        **explainer_profile.to_cf_params_for_row(row),
    )

The explainer profile's `to_cf_params()` method returns only the parameters
supported by the corresponding DiCE explainer.

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- DiCE explainers differ significantly in which parameters they accept.
- Passing unsupported parameters will raise TypeError at runtime.
- This module ensures that only valid parameters are forwarded to DiCE.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Directional bounds (your original definitions preserved)
# -------------------------------------------------------------------------

DIRECTIONAL_BOUNDS = {
    # feature: (direction, min_value, max_value)
    "bmi": ("down", 15, 40),
    "etfruit": ("down", 1, 7),  # inverted (1 = much, 6 = less)
    "eatveg": ("down", 1, 7),  # inverted
    "cgtsmok": ("up", 1, 6),  # inverted
    "alcfreq": ("up", 1, 7),  # inverted
    "slprl": ("down", 1, 4),
    "paccnois": ("down", 0, 1),
    "dosprt": ("up", 1, 7),
}


def build_directional_ranges(row, rules, ordinal_allowed_values):
    """
    Build permitted ranges based on directional rules.

    Ordinal features expand to full discrete sets within bounds.
    Continuous features keep simple [min, max] bounds.

    Validates that:
    - Current values are valid for ordinal features
    - Directional bounds stay within allowed values

    CRITICAL DTYPE HANDLING:
    - DiCE treats ordinal/categorical features as strings internally
    - This function returns STRING lists for ordinal features
    - Continuous features remain as [float, float] bounds
    - This ensures compatibility with DiCE's internal validation

    Parameters
    ----------
    row : pd.Series
        Current feature values for a single query instance.
        Ordinal features should be strings (as loaded by _load_data).
    rules : dict
        Directional rules mapping feature -> (direction, min_val, max_val)
    ordinal_allowed_values : dict
        Mapping of ordinal feature -> list of allowed integer values

    Returns
    -------
    dict
        Mapping of feature -> permitted range
        - Ordinal features: list of STRING values (e.g., ["1", "2", "3"])
        - Continuous features: [float, float] bounds (e.g., [15.0, 40.0])

    Raises
    ------
    ValueError
        If current value is invalid or directional range is empty
    """
    ranges = {}

    for feat, (direction, min_val, max_val) in rules.items():
        current = row[feat]

        # Validate and normalize ordinal current value
        if feat in ordinal_allowed_values:
            allowed = ordinal_allowed_values[feat]

            # Convert string to int for validation (handles DiCE string input)
            if isinstance(current, str):
                try:
                    current_int = int(current)
                except ValueError:
                    raise ValueError(
                        f"Cannot convert current value '{current}' to int "
                        f"for ordinal feature '{feat}'"
                    )
            else:
                # Handle float values (from XGBoost if not yet converted)
                current_int = int(round(current))

            if current_int not in allowed:
                raise ValueError(
                    f"Current value {current} (normalized to {current_int}) "
                    f"for feature '{feat}' is not in allowed values {allowed}"
                )

            current = current_int
            logger.debug(f"Validated ordinal feature '{feat}': current={current}")

        # Set direction bounds
        if direction == "up":
            lo, hi = current, max_val
        elif direction == "down":
            lo, hi = min_val, current
        else:
            raise ValueError(f"Unknown direction '{direction}' for feature '{feat}'")

        # Ordinal -> expand to all discrete values within bounds
        if feat in ordinal_allowed_values:
            lo_i = int(lo)
            hi_i = int(hi)

            # Filter to only valid values within bounds
            valid_range = [v for v in allowed if lo_i <= v <= hi_i]

            if not valid_range:
                raise ValueError(
                    f"Directional range [{lo_i}, {hi_i}] for feature '{feat}' "
                    f"contains no valid values from allowed set {allowed}"
                )

            # CRITICAL: Convert to strings for DiCE compatibility
            ranges[feat] = [str(v) for v in valid_range]
            logger.debug(
                f"Built ordinal range for '{feat}': {ranges[feat]} "
                f"(dtype: string for DiCE)"
            )

        # Continuous -> keep range as [min, max]
        else:
            ranges[feat] = [float(lo), float(hi)]
            logger.debug(f"Built continuous range for '{feat}': [{lo}, {hi}]")

    return ranges


# -------------------------------------------------------------------------
# Abstract Base Class
# -------------------------------------------------------------------------
@dataclass
class BaseExplainerProfile(ABC):
    """
    Base class for all explainer profiles.

    Subclasses must define:
        - supported_params: list[str]
        - method: str

    This class handles:
        - merging permitted ranges
        - building directional ranges
        - constructing final DiCE parameter dict
        - ensuring dtype consistency (strings for ordinals, floats for continuous)

    DTYPE CONSISTENCY STRATEGY:
    - Training data: ordinals are strings (loaded by _load_data)
    - Query instances: ordinals are strings (converted in dice_batch_runner)
    - Permitted ranges: ordinals are strings (ensured here)
    - This matches DiCE's internal string-based categorical handling
    """

    total_CFs: int
    stopping_threshold: float
    ordinal_allowed_values: Dict[str, List]

    # Common optional fields
    features_to_vary: Optional[List[str]] = None
    permitted_range: Optional[Dict] = None
    desired_class: int = 0
    use_permitted_range: bool = True  # Enable/disable permitted_range constraints

    @property
    @abstractmethod
    def supported_params(self) -> List[str]:
        """List of parameter names supported by this explainer."""
        pass

    @property
    @abstractmethod
    def method(self) -> str:
        """Name of the DiCE explainer method."""
        pass

    # ------------------------------------------------------------------
    # Shared logic for all explainers
    # ------------------------------------------------------------------
    def to_cf_params(self):
        """Return only the parameters supported by this explainer."""
        base = {
            "total_CFs": self.total_CFs,
            "desired_class": self.desired_class,
            "features_to_vary": self.features_to_vary,
            "permitted_range": self.permitted_range,
            "stopping_threshold": self.stopping_threshold,
        }

        # Filter to only supported params
        return {k: v for k, v in base.items() if k in self.supported_params}

    def to_cf_params_for_row(self, row):
        """
        Merge directional ranges with static permitted_range.

        Ensures dtype consistency for DiCE:
        - Ordinal features: list of STRINGS (e.g., ["1", "2", "3"])
        - Continuous features: [float, float] bounds

        DiCE performs strict equality checks between:
        1. Query instance values
        2. Training data values
        3. Permitted range values

        All three must have matching dtypes (strings for ordinals).

        Parameters
        ----------
        row : pd.Series
            Single query instance with string dtypes for ordinal features

        Returns
        -------
        dict
            DiCE parameters with consistent dtypes in permitted_range
        """
        params = self.to_cf_params().copy()

        # Only build permitted_range if enabled
        if not self.use_permitted_range:
            logger.info("Permitted range disabled - skipping constraint generation")
            # Remove permitted_range from params if it exists
            params.pop("permitted_range", None)
            return params

        # Build directional ranges (returns strings for ordinals)
        directional = build_directional_ranges(
            row, DIRECTIONAL_BOUNDS, self.ordinal_allowed_values
        )

        # Merge with any static permitted_range
        if self.permitted_range:
            merged = self.permitted_range.copy()
            merged.update(directional)
            params["permitted_range"] = merged
        else:
            params["permitted_range"] = directional

        # Ensure ordinal permitted ranges are STRINGS (for DiCE)
        # This is defensive - build_directional_ranges already returns strings,
        # but we ensure consistency in case permitted_range was set manually
        for feat, values in params["permitted_range"].items():
            if feat in self.ordinal_allowed_values:
                params["permitted_range"][feat] = [str(v) for v in values]
                logger.debug(
                    f"Ensured string dtype for permitted_range['{feat}']: "
                    f"{params['permitted_range'][feat]}"
                )

        # Fallback: ensure ALL ordinal features have permitted_range
        # (use full allowed set if not specified in directional bounds)
        for feat, allowed in self.ordinal_allowed_values.items():
            if feat not in params["permitted_range"]:
                params["permitted_range"][feat] = [str(v) for v in allowed]
                logger.debug(
                    f"Added fallback permitted_range for '{feat}': "
                    f"{params['permitted_range'][feat]}"
                )

        return params

    def __str__(self):
        header = f"=== {self.__class__.__name__} ==="
        lines = [f"{key:23}: {value}" for key, value in self.__dict__.items()]
        return header + "\n" + "\n".join(lines)


# -------------------------------------------------------------------------
# Random Explainer
# -------------------------------------------------------------------------
@dataclass
class RandomExplainerProfile(BaseExplainerProfile):
    posthoc_sparsity_param: float = 0.1
    posthoc_sparsity_algorithm: str = "linear"
    random_seed: int = 101

    @property
    def method(self):
        return "random"

    @property
    def supported_params(self):
        return [
            "total_CFs",
            "desired_class",
            "features_to_vary",
            "permitted_range",
            "stopping_threshold",
            "posthoc_sparsity_param",
            "posthoc_sparsity_algorithm",
            "random_seed",
        ]


# -------------------------------------------------------------------------
# Genetic Explainer
# -------------------------------------------------------------------------
@dataclass
class GeneticExplainerProfile(BaseExplainerProfile):
    posthoc_sparsity_param: float = 0.1
    posthoc_sparsity_algorithm: str = "linear"
    maxiterations: int = 1000
    verbose: bool = False

    @property
    def method(self):
        return "genetic"

    @property
    def supported_params(self):
        return [
            "total_CFs",
            "desired_class",
            "features_to_vary",
            "permitted_range",
            "stopping_threshold",
            "posthoc_sparsity_param",
            "posthoc_sparsity_algorithm",
            "maxiterations",
            "verbose",
        ]


# -------------------------------------------------------------------------
# Gradient Explainer
# -------------------------------------------------------------------------
@dataclass
class GradientExplainerProfile(BaseExplainerProfile):
    @property
    def method(self):
        return "gradientdescent"

    @property
    def supported_params(self):
        return [
            "total_CFs",
            "desired_class",
            "features_to_vary",
            "permitted_range",
        ]
