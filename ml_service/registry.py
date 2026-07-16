"""Read final-model metadata without exposing stage paths to backend callers."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .exceptions import InvalidHorizonError, ModelNotFoundError, RegistryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "models" / "legacy_registry" / "model_registry.json"


class ModelRegistry:
    """Provide validated final-model metadata from the single registry file."""

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self.path = path
        try:
            self.payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(self.payload["models"], dict):
                raise KeyError("models")
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RegistryNotFoundError(
                f"Final model registry is unavailable or invalid: {path}"
            ) from exc

    def get(self, horizon: str) -> dict[str, Any]:
        try:
            entry = self.payload["models"][horizon]
        except KeyError as exc:
            raise InvalidHorizonError(f"Unsupported horizon: {horizon}") from exc
        try:
            model_path = ROOT / entry["path"]
            entry["stage"]
            entry["feature_count"]
            entry["model_version"]
        except (KeyError, TypeError) as exc:
            raise RegistryNotFoundError(
                f"Final model registry entry is invalid for horizon: {horizon}"
            ) from exc
        if not model_path.is_file():
            raise ModelNotFoundError(f"Final model is missing: {model_path}")
        return dict(entry) | {"resolved_path": model_path}

    def info(self) -> dict[str, Any]:
        return {
            "registry_version": self.payload["version"],
            "models": self.payload["models"],
        }
