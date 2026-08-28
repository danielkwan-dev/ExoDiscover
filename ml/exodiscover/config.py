"""Project-wide settings.

The seed lives here and nowhere else: every estimator and every split reads
``settings.random_seed`` so a run is reproducible from this one value.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    root: Path = Path(__file__).resolve().parents[2]
    random_seed: int = 42
    cv_folds: int = 5
    #: Fraction of host stars held out of training entirely. Model selection
    #: and tuning run on the 70% via grouped CV; the 30% is touched once, to
    #: produce the reported numbers.
    test_size: float = 0.3

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.root / "models" / "production"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "docs" / "metrics"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / "metrics.json"


settings = Settings()
