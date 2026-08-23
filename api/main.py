"""ExoDiscover API.

Replaces the original 20-line Flask app, which ran with `debug=True`, loaded
the model on every request, performed no validation, and silently truncated
input to 2000 rows.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchResponse,
    KOIInput,
    LightCurveRequest,
    LightCurveStages,
    ModelInfo,
    Prediction,
)
from api.service import service
from exodiscover.config import settings
from exodiscover.features import lightcurve as lc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exodiscover.api")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 5_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    if service.available:
        logger.info("model %s loaded with %d features", service.version, len(service.features))
    else:
        logger.warning("no model artifact found; prediction endpoints will return 503")
    yield


app = FastAPI(
    title="ExoDiscover API",
    version="0.1.0",
    description=(
        "Calibrated exoplanet classification over the NASA Kepler KOI catalog. "
        "Every probability is isotonic-calibrated and every prediction carries "
        "its SHAP contributions."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_model() -> None:
    if not service.available:
        raise HTTPException(
            status_code=503,
            detail="No trained model is loaded. Run `exo train` to produce one.",
        )


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "model_loaded": service.available}


@app.get("/model-info", response_model=ModelInfo, tags=["meta"])
def model_info() -> dict:
    return service.model_info()


@app.get("/metrics", tags=["meta"])
def metrics() -> dict:
    """Full metrics payload backing the dashboard: ladder, ablations, transfer."""
    if not service.metrics:
        raise HTTPException(status_code=503, detail="No metrics available. Run `exo train`.")
    return service.metrics


@app.get("/discoveries", tags=["predict"])
def discoveries(limit: int = 25) -> dict:
    """Top-ranked unvetted KOI candidates, with the reasons behind each."""
    path = settings.metrics_dir / "top_candidates.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="No discovery ranking available.")
    df = pd.read_csv(path).head(max(1, min(limit, 200)))
    return {"n": len(df), "candidates": df.to_dict(orient="records")}


@app.post("/predict", response_model=Prediction, tags=["predict"])
def predict(payload: KOIInput) -> dict:
    _require_model()
    row = payload.model_dump()
    # build_features derives multiplicity from kepid; the caller supplies the
    # count directly, so synthesise a frame that reproduces it.
    count = int(row.pop("n_kois_on_star", 1))
    frame = pd.DataFrame([{**row, "kepid": 1}] * count)
    return service.predict_frame(frame)[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
async def predict_batch(file: UploadFile = File(...)) -> dict:
    _require_model()

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 5 MB limit.")

    try:
        df = pd.read_csv(io.BytesIO(contents), comment="#", low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV contains no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV has {len(df)} rows; the limit is {MAX_BATCH_ROWS}.",
        )

    required = {"koi_period", "koi_depth", "koi_duration", "koi_prad", "koi_srad", "koi_slogg"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV missing columns: {', '.join(missing)}")

    results = service.predict_frame(df, top_n=3)
    results.sort(key=lambda r: r["probability"], reverse=True)
    return {"n_rows": len(results), "results": results}


@app.post("/lightcurve/preprocess", response_model=LightCurveStages, tags=["lightcurve"])
def lightcurve_preprocess(payload: LightCurveRequest) -> dict:
    """Run the four preprocessing stages, for the 'how it works' explainer."""
    import numpy as np

    stages = lc.preprocess(np.asarray(payload.flux, dtype=float))
    flattened = stages["flattened"]
    return {
        "normalized": stages["normalized"].tolist(),
        "flattened": flattened.tolist(),
        "n_windows": int(len(stages["windows"])),
        "features": lc.window_features(flattened),
    }
