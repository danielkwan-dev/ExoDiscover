"""API contract tests.

These run against a model trained on the fixture, so they exercise the real
prediction path rather than a mock.
"""

import io

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.service import band_for, service
from exodiscover.config import settings
from exodiscover.features.tabular import FEATURE_COLUMNS, build_features
from exodiscover.models.registry import candidates

VALID_KOI = {
    "koi_period": 0.837495,
    "koi_depth": 152.0,
    "koi_duration": 1.811,
    "koi_prad": 1.47,
    "koi_srad": 1.065,
    "koi_slogg": 4.35,
    "koi_steff": 5627.0,
    "koi_impact": 0.30,
    "koi_model_snr": 25.0,
    "n_kois_on_star": 2,
}


@pytest.fixture
def client(tmp_path, koi_sample, monkeypatch):
    """A client backed by a real model fitted on the fixture.

    The settings root is redirected at tmp_path so the app's own lifespan
    handler discovers this bundle. Loading it here directly would not work:
    startup calls `service.load()` with no arguments and would overwrite it
    with whatever is in models/production.
    """
    monkeypatch.setattr(settings, "root", tmp_path)
    models_dir = tmp_path / "models" / "production"
    models_dir.mkdir(parents=True)

    df = koi_sample[
        koi_sample["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
    ].reset_index(drop=True)
    X = build_features(df)
    y = (df["koi_disposition"] == "CONFIRMED").astype(int)
    model = candidates(42)["lightgbm"].fit(X, y)

    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS, "version": "test"},
        models_dir / "model.joblib",
    )

    service.__init__()
    with TestClient(app) as c:
        yield c
    service.__init__()


def test_health_is_always_available(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_reports_the_feature_list(client):
    body = client.get("/model-info").json()
    assert body["available"] is True
    assert body["features"] == FEATURE_COLUMNS


def test_predict_returns_a_calibrated_probability_and_reasons(client):
    body = client.post("/predict", json=VALID_KOI).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in {"planet", "false positive"}
    assert body["band"] in {"confirmed", "needs vetting", "false positive"}
    assert len(body["contributions"]) > 0
    assert {"feature", "value", "shap"} == set(body["contributions"][0])


def test_predict_rejects_a_negative_period(client):
    response = client.post("/predict", json={**VALID_KOI, "koi_period": -1.0})
    assert response.status_code == 422
    assert "koi_period" in response.text


def test_predict_rejects_a_missing_required_field(client):
    payload = {k: v for k, v in VALID_KOI.items() if k != "koi_depth"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_ranks_results_by_probability(client, koi_sample):
    buf = io.BytesIO()
    koi_sample.head(30).to_csv(buf, index=False)
    buf.seek(0)
    body = client.post(
        "/predict/batch", files={"file": ("koi.csv", buf, "text/csv")}
    ).json()
    probs = [r["probability"] for r in body["results"]]
    assert probs == sorted(probs, reverse=True)
    assert body["n_rows"] == 30


def test_batch_rejects_a_csv_missing_required_columns(client):
    buf = io.BytesIO()
    pd.DataFrame({"nonsense": [1, 2]}).to_csv(buf, index=False)
    buf.seek(0)
    response = client.post("/predict/batch", files={"file": ("bad.csv", buf, "text/csv")})
    assert response.status_code == 422
    assert "koi_period" in response.json()["detail"]


def test_lightcurve_preprocess_returns_every_stage(client):
    flux = ([1.0] * 100 + [0.99] * 10) * 4
    body = client.post("/lightcurve/preprocess", json={"flux": flux}).json()
    assert len(body["normalized"]) == len(flux)
    assert "num_dips" in body["features"]
    assert body["n_windows"] >= 0


def test_lightcurve_rejects_a_series_that_is_too_short(client):
    assert client.post("/lightcurve/preprocess", json={"flux": [1.0, 2.0]}).status_code == 422


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.05, "false positive"), (0.5, "needs vetting"), (0.95, "confirmed")],
)
def test_bands_partition_the_probability_range(probability, expected):
    assert band_for(probability) == expected


def test_endpoints_return_503_when_no_model_is_loaded():
    service.__init__()  # unloaded state
    with TestClient(app) as c:
        c.app.dependency_overrides = {}
        service.__init__()
        assert c.post("/predict", json=VALID_KOI).status_code == 503
