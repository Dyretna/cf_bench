from abc import ABC
from pathlib import Path

import joblib


class LoaderTemplate(ABC):
    SUPPORTED_TYPES = []  # Must be overridden in subclasses

    def _validate_path(self, path):
        file_path = Path(path) if isinstance(path, str) else path
        if not isinstance(file_path, Path):
            raise TypeError("Loader accepts instances of 'pathlib.Path' or 'str'")

        if not file_path.exists():
            raise ValueError("Path to model is not a valid path")
        if not file_path.is_file():
            raise FileNotFoundError("File not found - check path")

        return file_path

    def _validate_type(self, path, supported_types):
        """Shared validation logic for file type checking."""
        if path.suffix.lower() not in supported_types:
            raise TypeError(
                f"Unsupported file format: {path.suffix}\n"
                f"Supported formats: {supported_types}"
            )

    def load(self, path, **kwargs): ...


class SKLearnModelLoader(LoaderTemplate):
    SUPPORTED_TYPES = [".pkl", ".joblib"]

    def load(self, path):
        """Load sklearn-compatible model (RandomForest, XGBoost, etc.).

        Args:
            path: Path to model file (.pkl or .joblib)

        Returns:
            dict: Contains 'model' and 'is_keras' (False)
        """
        try:
            file_path = self._validate_path(path)
            self._validate_type(file_path, self.SUPPORTED_TYPES)

            model = joblib.load(file_path)

            return {"model": model, "is_keras": False, "scaler": None}

        except Exception as e:
            print(f"Caught an exception while loading sklearn model:\n{e}")
            raise


class KerasModelLoader(LoaderTemplate):
    SUPPORTED_TYPES = [".keras", ".h5"]

    def load(self, path, scaler_path=None):
        """Load Keras/TensorFlow model and optional scaler.

        Args:
            path: Path to Keras model file (.keras or .h5)
            scaler_path: Optional path to scaler file (.pkl or .joblib)

        Returns:
            dict: Contains 'model', 'is_keras' (True), and optional 'scaler'
        """
        try:
            file_path = self._validate_path(path)
            self._validate_type(file_path, self.SUPPORTED_TYPES)

            import tensorflow as tf

            model = tf.keras.models.load_model(file_path)

            scaler = None
            if scaler_path:
                scaler_file = self._validate_path(scaler_path)
                scaler = joblib.load(scaler_file)

            return {"model": model, "is_keras": True, "scaler": scaler}

        except Exception as e:
            print(f"Caught an exception while loading Keras model:\n{e}")
            raise


def load_model_by_backend(backend, model_path, scaler_path=None):
    """Factory function to load model and scaler based on backend type.

    Args:
        backend: Backend type ('TF2' for Keras, 'sklearn' for sklearn models)
        model_path: Path to model file
        scaler_path: Optional path to scaler file (for Keras models)

    Returns:
        tuple: (model, is_keras, scaler)
    """
    if backend == "TF2":
        # Load Keras model with scaler
        keras_loader = KerasModelLoader()
        result = keras_loader.load(model_path, scaler_path=scaler_path)
    else:
        # Load sklearn model (RandomForest, XGBoost)
        sklearn_loader = SKLearnModelLoader()
        result = sklearn_loader.load(model_path)

    return result["model"], result["is_keras"], result["scaler"]
