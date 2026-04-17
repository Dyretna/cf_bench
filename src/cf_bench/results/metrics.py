"""Performance metrics calculation and formatting."""

from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


class PerformanceMetrics:
    """Calculate and format model performance metrics."""

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, any]:
        report = classification_report(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        return {"classification_report": report, "roc_auc": roc_auc}

    @staticmethod
    def compute_timing_metrics(cf_times: List[float]) -> Dict[str, float]:
        if not cf_times:
            return {
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
                "num_instances": 0,
            }

        return {
            "total_time": sum(cf_times),
            "avg_time": sum(cf_times) / len(cf_times),
            "min_time": min(cf_times),
            "max_time": max(cf_times),
            "num_instances": len(cf_times),
        }

    @staticmethod
    def format_timing_metrics(timing_metrics: Dict[str, float]) -> str:
        return (
            f"Total CF generation time: {timing_metrics['total_time']:.2f}s\n"
            f"Average time per instance: {timing_metrics['avg_time']:.2f}s\n"
            f"Min time: {timing_metrics['min_time']:.2f}s\n"
            f"Max time: {timing_metrics['max_time']:.2f}s\n"
            f"Number of instances: {timing_metrics['num_instances']}\n"
        )
