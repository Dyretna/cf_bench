"""Result export utilities and strategies."""

import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from .metrics import PerformanceMetrics

if TYPE_CHECKING:
    from ..config import BaseExplainerProfile, SystemConfig


def create_output_directory(
    output_base: str,
    model_type: str,
    explainer_method: str,
    use_permitted_range: bool,
    threshold: float,
    run_id: str = None,
) -> Path:
    """Create a timestamped output directory for pipeline results."""

    if use_permitted_range:
        prange = "prange"
    else:
        prange = ""

    if 0 < threshold <= 0.3:
        thres = "low"
    elif 0.4 <= threshold <= 0.6:
        thres = "mid"
    elif 0.7 < threshold < 1:
        thres = "high"
    else:
        thres = "unknown"

    today = dt.datetime.today().strftime("%Y-%m-%d")
    run_name = f"{model_type}_{explainer_method}_{prange}_{thres}thres_{today}"
    if run_id:
        run_name += f"_{run_id}"
    output_dir = Path(output_base) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class ConfigExporter:
    """Export configuration and timing information."""

    @staticmethod
    def export(
        output_path: Path,
        system_config: "SystemConfig",
        explainer_profile: "BaseExplainerProfile",
        timing_metrics: Dict[str, float],
    ) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=== CONFIGURATION ===\n\n")
            f.write(str(system_config) + "\n\n")
            f.write(str(explainer_profile))
            f.write("\n\n=== TIMING INFORMATION ===\n\n")
            f.write(PerformanceMetrics.format_timing_metrics(timing_metrics))


class ModelInfoExporter:
    """Export model information and performance metrics."""

    @staticmethod
    def export(output_path: Path, model_info: Dict, performance_metrics: Dict) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=== MODEL INFO ===\n\n")
            f.write(f"Model Type: {model_info['model_type']}\n")
            f.write(f"Model Class: {model_info['model_class']}\n")
            f.write(f"Model Module: {model_info['model_module']}\n\n")

            # Handle Keras models with summary
            if "summary" in model_info:
                f.write("Model Summary:\n")
                f.write(model_info["summary"])
                f.write(f"\nNumber of Layers: {model_info.get('num_layers', 'N/A')}\n")
                f.write(f"Total Parameters: {model_info.get('total_params', 'N/A')}\n")
            else:
                # For non-Keras models, write parameters
                f.write("Parameters:\n")
                params = model_info.get("params", {})
                for k, v in params.items():
                    f.write(f"  {k}: {v}\n")

            # Add feature importances if available
            if "feature_importances" in model_info:
                f.write("\nFeature Importances:\n")
                # Sort by importance (descending)
                sorted_features = sorted(
                    model_info["feature_importances"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for feat, importance in sorted_features:
                    f.write(f"  {feat}: {importance:.4f}\n")

            f.write("\n=== PERFORMANCE ===\n")
            f.write(performance_metrics["classification_report"])
            f.write(f"\nROC-AUC: {performance_metrics['roc_auc']:.4f}\n")

    @staticmethod
    def export_json(output_path: Path, model_info: Dict) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, default=str)
