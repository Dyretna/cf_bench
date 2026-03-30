import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR"))
MODELS_DIR = Path(os.getenv("MODELS_DIR"))
CF_OUTPUTS_DIR = Path(os.getenv("CF_OUTPUTS"))

print("data dir path ok:        ", DATA_DIR.is_dir())
print("model dir path ok:       ", MODELS_DIR.is_dir())
print("CF outputs dir path ok:  ", CF_OUTPUTS_DIR.is_dir())


@dataclass
class DicePipelineConfig:
    target_bp: str = "hltprhb"
    target_hc: str = "hltprhc"

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

    train_data_path_bp: Path = field(init=False)
    train_data_path_hc: Path = field(init=False)
    test_data_path_bp: Path = field(init=False)
    test_data_path_hc: Path = field(init=False)

    model_path_bp: Path = field(init=False)
    model_path_hc: Path = field(init=False)

    def __post_init__(self):
        self.features_to_vary = [
            c for c in self.feature_cols if c not in self.immutable_cols
        ]

        self.cf_output_dir: Path = CF_OUTPUTS_DIR
        self.train_data_path_bp = (
            DATA_DIR / "05_single_target" / f"ess_ready_v2_{self.target_bp}_train.csv"
        )
        self.train_data_path_hc = (
            DATA_DIR / "05_single_target" / f"ess_ready_v2_{self.target_hc}_train.csv"
        )
        self.test_data_path_bp = (
            DATA_DIR / "05_single_target" / f"ess_ready_v2_{self.target_bp}_test.csv"
        )
        self.test_data_path_hc = (
            DATA_DIR / "05_single_target" / f"ess_ready_v2_{self.target_hc}_test.csv"
        )

        self.model_path_bp = MODELS_DIR / f"rf_{self.target_bp}_2026-03-26.pkl"
        self.model_path_hc = MODELS_DIR / f"rf_{self.target_hc}_2026-03-26.pkl"

    def __str__(self):
        lines = [f"{key:23}: {value}" for key, value in self.__dict__.items()]

        return "\n".join(lines)
