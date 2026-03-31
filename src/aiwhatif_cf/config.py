import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR"))
MODELS_DIR = Path(os.getenv("MODELS_DIR"))
CF_OUTPUTS_DIR = Path(os.getenv("CF_OUTPUTS"))


@dataclass
class DicePipelineConfig:
    target: str
    backend: str = "sklearn"
    model_type: str = "classifier"
    explainer_method: str = "random"

    stopping_threshold: float = 0.5
    posthoc_sparsity_param: float = 0.1

    target_factor: float = 0.5
    cf_random_seed: int = 42

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

    # Paths optional - defaults are set in __post_init__
    train_data_path: Optional[Path] = None
    test_data_path: Optional[Path] = None
    model_path: Optional[Path] = None
    cf_output_path: Optional[Path] = None

    def __post_init__(self):
        self.features_to_vary = [
            c for c in self.feature_cols if c not in self.immutable_cols
        ]

        # Path defaults
        if self.train_data_path is None:
            self.train_data_path = (
                DATA_DIR / "05_single_target" / f"ess_ready_v2_{self.target}_train.csv"
            )

        if self.test_data_path is None:
            self.test_data_path = (
                DATA_DIR / "05_single_target" / f"ess_ready_v2_{self.target}_test.csv"
            )

        if self.model_path is None:
            self.model_path = MODELS_DIR / f"rf_{self.target}_2026-03-26.pkl"

        if self.cf_output_path is None:
            self.cf_output_path = CF_OUTPUTS_DIR

    def __str__(self):
        lines = [f"{key:23}: {value}" for key, value in self.__dict__.items()]

        return "\n".join(lines)
