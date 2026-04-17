"""Data loading strategies for DiCE and model compatibility."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..config import SystemConfig


class DataLoader(ABC):
    """Base class for data loading strategies."""

    @abstractmethod
    def load(self, path: str, config: "SystemConfig") -> pd.DataFrame:
        """
        Load data from file with appropriate dtype handling.

        Parameters
        ----------
        path : str
            Path to CSV file
        config : SystemConfig
            Configuration object containing feature metadata

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame with correct dtypes
        """
        pass


class DiCECompatibleLoader(DataLoader):
    """
    Load data with DiCE-compatible dtypes.

    DiCE treats ordinal/categorical features as strings internally and performs
    strict equality checks. To ensure compatibility:
    - Ordinal features are loaded as strings
    - Continuous features are loaded as floats

    This ensures that DiCE's internal validation passes and that permitted_range
    values match training data dtypes exactly.
    """

    def load(self, path: str, config: "SystemConfig") -> pd.DataFrame:
        """Load data with DiCE-compatible dtypes."""
        dtype_map = {
            **{col: str for col in config.ordinal_features},  # strings for DiCE
            **{col: float for col in config.continuous_features},
        }

        df = pd.read_csv(path, dtype=dtype_map)

        # Remove any unnamed columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Ensure correct column order: features + target
        df = df[config.feature_cols + [config.target]]

        return df


def load_dice_compatible_data(path: str, config: "SystemConfig") -> pd.DataFrame:
    """Convenience function to load data with DiCE-compatible dtypes."""
    loader = DiCECompatibleLoader()
    return loader.load(path, config)
