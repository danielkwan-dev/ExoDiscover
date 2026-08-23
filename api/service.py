"""Model loading and scoring.

The model is loaded once at process start rather than per request. Prediction
bands come from the calibrated probability, which is what lets a binary
classifier present a three-class vocabulary honestly: the middle band is
"the model is not confident either way", not a learned third category.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from exodiscover.config import settings
from exodiscover.explain import explain_rows
from exodiscover.features.tabular import build_features

#: Probability below which a signal reads as a false positive, and above which
#: it reads as confirmed. Between them the model is declaring uncertainty.
FALSE_POSITIVE_BELOW = 0.35
CONFIRMED_ABOVE = 0.75


def band_for(probability: float) -> str:
    if probability < FALSE_POSITIVE_BELOW:
        return "false positive"
    if probability >= CONFIRMED_ABOVE:
        return "confirmed"
    return "needs vetting"


class ModelService:
    """Holds the loaded bundle. Degrades to `available = False` rather than
    crashing the process when no model has been trained yet."""

    def __init__(self) -> None:
        self.model: Any = None
        self.features: list[str] = []
        self.version: str = "unknown"
        self.metrics: dict = {}

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self, model_path: Path | None = None, metrics_path: Path | None = None) -> None:
        path = model_path or (settings.models_dir / "model.joblib")
        if path.exists():
            bundle = joblib.load(path)
            self.model = bundle["model"]
            self.features = list(bundle["features"])
            self.version = bundle.get("version", "unknown")

        metrics = metrics_path or settings.metrics_path
        if metrics.exists():
            self.metrics = json.loads(metrics.read_text())

    def _require(self) -> None:
        if not self.available:
            raise RuntimeError(
                "no trained model available; run `exo train` to produce "
                "models/production/model.joblib"
            )

    def predict_frame(self, df: pd.DataFrame, *, top_n: int = 5) -> list[dict]:
        """Score a frame of raw KOI rows."""
        self._require()
        X = build_features(df)[self.features]
        probabilities = self.model.predict_proba(X)[:, 1]
        contributions = explain_rows(self.model, X, top_n=top_n)

        return [
            {
                "row": i,
                "probability": float(p),
                "label": "planet" if p >= 0.5 else "false positive",
                "band": band_for(float(p)),
                "contributions": contributions[i],
            }
            for i, p in enumerate(probabilities)
        ]

    def model_info(self) -> dict:
        block = self.metrics.get("model", {})
        return {
            "name": block.get("name", "unknown"),
            "version": self.version,
            "trained_at": block.get("trained_at", "unknown"),
            "framing": block.get("framing", "binary"),
            "features": self.features,
            "test": self.metrics.get("test", {}),
            "available": self.available,
        }


service = ModelService()
