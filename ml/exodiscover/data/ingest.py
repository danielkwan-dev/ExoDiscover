"""Client for the NASA Exoplanet Archive TAP service.

Replaces the four ad-hoc download scripts in the original backend. Every
fetch records its query text and timestamp, so a training run can state
exactly which archive snapshot produced it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from exodiscover.config import settings

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

TABLES: dict[str, str] = {
    "koi": "cumulative",
    "toi": "toi",
    "k2": "k2pandc",
    # Gaia-derived distances per star, joined to the KOI catalog on kepid. The
    # KOI table has ra and dec but nothing to place an object at a distance.
    "stellar": "Q1_Q17_DR25_KS",
}

#: Columns to request, for tables where fetching everything is wasteful.
#: Q1_Q17_DR25_KS is 200,038 rows of 99 columns; four of them are wanted, and
#: `select *` would transfer roughly 200 MB to use under 6 MB of it.
COLUMNS: dict[str, tuple[str, ...]] = {
    "stellar": ("kepid", "dist", "dist_err1", "dist_err2"),
}


def build_query(table: str) -> str:
    if table not in TABLES:
        raise KeyError(f"unknown table {table!r}; expected one of {sorted(TABLES)}")
    selected = ", ".join(COLUMNS[table]) if table in COLUMNS else "*"
    return f"select {selected} from {TABLES[table]}"


def _download(query: str) -> pd.DataFrame:
    response = requests.get(TAP_URL, params={"query": query, "format": "csv"}, timeout=600)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), comment="#", low_memory=False)


def _paths(table: str) -> tuple[Path, Path]:
    raw = settings.root / "data" / "raw"
    return raw / f"{table}.csv", raw / f"{table}.meta.json"


def fetch(table: str, *, force: bool = False) -> pd.DataFrame:
    """Return the catalog, downloading it only if not already cached."""
    csv_path, meta_path = _paths(table)
    if csv_path.exists() and not force:
        return pd.read_csv(csv_path, comment="#", low_memory=False)

    query = build_query(table)
    df = _download(query)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "query": query,
                "fetched_at": datetime.now(UTC).isoformat(),
                "n_rows": int(len(df)),
            },
            indent=2,
        )
    )
    return df


def load_cached(table: str) -> pd.DataFrame:
    csv_path, _ = _paths(table)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} missing; run `exo ingest` first")
    return pd.read_csv(csv_path, comment="#", low_memory=False)
