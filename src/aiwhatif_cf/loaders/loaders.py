from abc import ABC
from pathlib import Path


class LoaderTemplate(ABC):
    SUPPORTED_TYPES = []  # Must be overridden in subclasses

    def _validate_path(path):
        file_path = Path(path) if isinstance(path, str) else path
        if not isinstance(file_path, Path):
            raise TypeError("Loader accepts instances of 'pathlib.Path' or 'str'")

        if not file_path.exists():
            raise ValueError("Path to image is not a valid path")
        if not file_path.is_file():
            raise FileNotFoundError("File not found - check path")

        return file_path

    def _validate_type(path, supported_types):
        """Shared validation logic for file type checking."""
        if path.suffix.lower() not in supported_types:
            raise TypeError(
                f"Unsupported file format: {path.suffix}\n"
                f"Supported formats: {supported_types}"
            )

    def load(path, **kwargs): ...


class CSVLoader(LoaderTemplate):
    SUPPORTED_TYPES = [".csv"]

    def load(self, path):
        """Validates path and CSV, then returns the path."""
        try:
            file_path = self._validate_path(path)
            self._validate_type(file_path, self.SUPPORTED_TYPES)
            return file_path
        except Exception as e:
            print(f"caught an exception: \n{e}")
