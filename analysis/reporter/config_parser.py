"""
Configuration parser for extracting experiment parameters from config text files.

Each experiment directory contains a config_*.txt file with all parameters used
during counterfactual generation. This module extracts these parameters so we can
properly analyze results without inferring them from file paths.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration parameters extracted from config file."""

    # Feature configuration
    feature_cols: list[str]
    features_to_vary: list[str]
    locked_features: list[str]

    # DiCE parameters
    stopping_threshold: float
    posthoc_sparsity_param: float
    total_cfs: int
    explainer_profile: str  # 'genetic', 'random', etc.
    maxiterations: int

    # Model information
    ml_model_type: str  # Extracted from model_info.json

    # Optional parameters
    posthoc_sparsity_algorithm: Optional[str] = None
    use_permitted_range: Optional[bool] = None


class ConfigParser:
    """Parse experiment configuration from text files."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.content = self._read_file()

    def _read_file(self) -> str:
        """Read config file content."""
        with open(self.config_path, "r") as f:
            return f.read()

    def _extract_list_value(self, key: str) -> Optional[list]:
        """
        Extract a list value from config file.

        Example line: "features_to_vary       : ['etfruit', 'eatveg', 'cgtsmok']"
        """
        pattern = rf"{key}\s*:\s*(\[.*?\])"
        match = re.search(pattern, self.content, re.MULTILINE)

        if not match:
            return None

        try:
            # Use ast.literal_eval to safely parse the list string
            return ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            return None

    def _extract_numeric_value(self, key: str) -> Optional[float]:
        """
        Extract a numeric value from config file.

        Example line: "stopping_threshold     : 0.1"
        """
        pattern = rf"{key}\s*:\s*([\d.]+)"
        match = re.search(pattern, self.content, re.MULTILINE)

        if not match:
            return None

        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _extract_int_value(self, key: str) -> Optional[int]:
        """Extract an integer value from config file."""
        value = self._extract_numeric_value(key)
        return int(value) if value is not None else None

    def _extract_bool_value(self, key: str) -> Optional[bool]:
        """
        Extract a boolean value from config file.

        Example line: "use_permitted_range    : True"
        """
        pattern = rf"{key}\s*:\s*(True|False)"
        match = re.search(pattern, self.content, re.MULTILINE)

        if not match:
            return None

        return match.group(1) == "True"

    def _extract_string_value(self, key: str) -> Optional[str]:
        """
        Extract a string value from config file.

        Example line: "posthoc_sparsity_algorithm: linear"
        """
        pattern = rf"{key}\s*:\s*(\w+)"
        match = re.search(pattern, self.content, re.MULTILINE)

        return match.group(1) if match else None

    def _extract_explainer_profile(self) -> str:
        """Extract explainer profile from section headers like 'GeneticExplainerProfile' or 'RandomSearchProfile'."""
        # Look for common patterns in config files
        if (
            "GeneticExplainerProfile" in self.content
            or "genetic" in self.config_path.name.lower()
        ):
            return "genetic"
        elif (
            "RandomSearchProfile" in self.content
            or "random" in self.config_path.name.lower()
        ):
            return "random"
        else:
            return "unknown"

    def parse(self) -> ExperimentConfig:
        """
        Parse config file and return ExperimentConfig object.

        Locked features are computed as: feature_cols - features_to_vary
        """
        # Extract required parameters
        feature_cols = self._extract_list_value("feature_cols")
        features_to_vary = self._extract_list_value("features_to_vary")
        stopping_threshold = self._extract_numeric_value("stopping_threshold")
        posthoc_sparsity_param = self._extract_numeric_value("posthoc_sparsity_param")
        total_cfs = self._extract_int_value("total_CFs")
        explainer_profile = self._extract_explainer_profile()
        maxiterations = self._extract_int_value("maxiterations")

        # Validate required parameters
        if feature_cols is None:
            raise ValueError(
                f"Could not extract 'feature_cols' from {self.config_path}"
            )
        if features_to_vary is None:
            raise ValueError(
                f"Could not extract 'features_to_vary' from {self.config_path}"
            )
        if stopping_threshold is None:
            raise ValueError(
                f"Could not extract 'stopping_threshold' from {self.config_path}"
            )
        if posthoc_sparsity_param is None:
            # Default value if not found
            posthoc_sparsity_param = 0.0
        if total_cfs is None:
            # Try alternative name
            total_cfs = self._extract_int_value("total_cfs")
            if total_cfs is None:
                raise ValueError(
                    f"Could not extract 'total_CFs' from {self.config_path}"
                )
        if maxiterations is None:
            maxiterations = 1000  # Default value

        # Compute locked features
        locked_features = [f for f in feature_cols if f not in features_to_vary]

        # Extract ML model type from model_info.json in same directory
        ml_model_type = extract_ml_model_type(self.config_path.parent)

        # Extract optional parameters
        posthoc_sparsity_algorithm = self._extract_string_value(
            "posthoc_sparsity_algorithm"
        )
        use_permitted_range = self._extract_bool_value("use_permitted_range")

        return ExperimentConfig(
            feature_cols=feature_cols,
            features_to_vary=features_to_vary,
            locked_features=locked_features,
            stopping_threshold=stopping_threshold,
            posthoc_sparsity_param=posthoc_sparsity_param,
            total_cfs=total_cfs,
            explainer_profile=explainer_profile,
            maxiterations=maxiterations,
            ml_model_type=ml_model_type,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
            use_permitted_range=use_permitted_range,
        )


def find_config_file(experiment_dir: Path) -> Optional[Path]:
    """
    Find config file in experiment directory.

    Looks for files matching pattern: config_*.txt or config*.txt
    """
    experiment_dir = Path(experiment_dir)

    # Try common patterns
    patterns = ["config_*.txt", "config*.txt"]

    for pattern in patterns:
        matches = list(experiment_dir.glob(pattern))
        if matches:
            return matches[0]  # Return first match

    return None


def extract_ml_model_type(experiment_dir: Path) -> str:
    """
    Extract ML model type from model info files (JSON or TXT).

    Flexible pattern matching:
    - model_info.json or model_*info.json (e.g., model_hltprhc_info.json)
    - model_info.txt or model_*info.txt (e.g., model_hltprhc_info.txt)
    - Fallback to experiment directory name if files don't contain explicit type

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Model type from file: 'XGBoost', 'RandomForest', 'NeuralNetwork', or 'unknown'
    """
    import json

    experiment_dir = Path(experiment_dir)

    # Try to find JSON files first (most reliable)
    # Pattern: model_info.json or model_*info.json
    json_files = list(experiment_dir.glob("model*info.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                model_type = data.get("model_type", "unknown")
                if model_type and model_type != "unknown":
                    return model_type
        except Exception as e:
            print(f"Warning: Could not parse {json_file.name} in {experiment_dir}: {e}")

    # Try TXT files as fallback
    # Pattern: model_info.txt or model_*info.txt
    txt_files = list(experiment_dir.glob("model*info.txt"))
    for txt_file in txt_files:
        try:
            with open(txt_file, "r") as f:
                content = f.read()
                # Look for "Model Type: XGBoost" or "Model Type: RandomForest" pattern
                match = re.search(r"Model Type:\s*(\w+)", content)
                if match:
                    return match.group(1)

                # Alternative: look for XGBoost-specific parameters
                if "objective: binary:logistic" in content or "booster:" in content:
                    return "XGBoost"

                # Alternative: look for sklearn RandomForest indicators
                if "RandomForestClassifier" in content or "n_estimators:" in content:
                    return "RandomForest"

        except Exception as e:
            print(f"Warning: Could not parse {txt_file.name} in {experiment_dir}: {e}")

    # Last resort: infer from directory name
    dir_name = experiment_dir.name.lower()
    if "xgb" in dir_name or "xgboost" in dir_name:
        return "XGBoost"
    elif "rf" in dir_name or "randomforest" in dir_name:
        return "RandomForest"
    elif "nn" in dir_name or "neural" in dir_name:
        return "NeuralNetwork"

    print(f"Warning: No model info files found in {experiment_dir}")
    return "unknown"
