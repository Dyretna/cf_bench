"""
Predictions module for aiwhatif_cf.

Exposes:
- PredictionService: huvudklassen för prediktioner
- Visualiseringsfunktioner för att plotta resultat
"""

from .prediction_service import PredictionService
from .visualizations import MultiClassifierVisualizations

__all__ = ["PredictionService", "MultiClassifierVisualizations"]
