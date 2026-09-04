"""Builds the sky-map artifact: where each catalogued object actually is.

Two missions, one file. That combination is the whole point of the view.

Kepler stared at a single 22x16 degree window for four years, so its 9,564
objects form a narrow beam. TESS surveyed the entire sky, so its objects sit in
every direction, and the nearest is 21 light years away rather than 391. Put
both in one scene and the difference between a staring mission and an all-sky
survey is visible rather than described -- and an observer at Earth is
genuinely surrounded, which is true of TESS and would be a lie about Kepler
alone.

Neither catalog carries a distance. Kepler's comes from the stellar table
(`Q1_Q17_DR25_KS`, joined on kepid) and TESS carries `st_dist` directly. Both
contain zeros and nulls, and neither is a distance: a row that cannot be placed
is dropped rather than imputed, because substituting a median would draw a
planet somewhere it is not.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from exodiscover.data.schema import TOI_LABEL_MAP
from exodiscover.features.tabular import add_multiplicity, build_features

#: Parsecs to light years.
LY_PER_PARSEC = 3.261563777

SKYMAP_COLUMNS: list[str] = [
    "name",
    "mission",
    "star_id",
    "ra",
    "dec",
    "dist_pc",
    "disposition",
    "probability",
    "radius_earth",
    "period_days",
    "top_reasons",
]

#: TESS column -> the KOI name the feature builder expects.
TOI_RENAME: dict[str, str] = {
    "pl_orbper": "koi_period",
    "pl_trandep": "koi_depth",
    "pl_trandurh": "koi_duration",
    "pl_rade": "koi_prad",
    "pl_insol": "koi_insol",
    "pl_eqt": "koi_teq",
    "st_teff": "koi_steff",
    "st_logg": "koi_slogg",
    "st_rad": "koi_srad",
    "st_tmag": "koi_kepmag",
    "tid": "kepid",
}

#: The two archives use different words for the same three states.
DISPOSITION = {2: "CONFIRMED", 1: "CANDIDATE", 0: "FALSE POSITIVE"}

Scorer = Callable[[pd.DataFrame], np.ndarray]
Reasoner = Callable[[pd.DataFrame], list[str]]


def usable_distances(stellar: pd.DataFrame) -> pd.DataFrame:
    """One row per star, keeping only distances that mean something."""
    out = stellar.dropna(subset=["dist"])
    out = out[out["dist"] > 0].drop_duplicates("kepid")
    return pd.DataFrame({"kepid": out["kepid"], "dist_pc": out["dist"]})


def _kepler_frame(koi: pd.DataFrame, stellar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counted = add_multiplicity(koi)
    features = build_features(counted)

    frame = pd.DataFrame(
        {
            "name": counted.get("kepoi_name"),
            "mission": "Kepler",
            "star_id": counted["kepid"],
            "ra": pd.to_numeric(counted.get("ra"), errors="coerce"),
            "dec": pd.to_numeric(counted.get("dec"), errors="coerce"),
            "disposition": counted.get("koi_disposition"),
            "radius_earth": pd.to_numeric(counted.get("koi_prad"), errors="coerce"),
            "period_days": pd.to_numeric(counted.get("koi_period"), errors="coerce"),
        }
    )
    merged = frame.merge(
        usable_distances(stellar).rename(columns={"kepid": "star_id"}),
        on="star_id",
        how="inner",
    )
    return merged, features.loc[merged.index.intersection(features.index)]


def _tess_frame(toi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every TOI, not only the resolved ones.

    `transfer.toi_to_common` keeps only CP/KP/FP/FA because those are the rows
    that carry ground truth to score against. Here the unvetted PC and APC rows
    are the interesting ones -- they are what the model has an opinion about
    and nobody has confirmed.
    """
    renamed = toi.rename(columns=TOI_RENAME).reset_index(drop=True)
    features = build_features(renamed)

    label = renamed.get("tfopwg_disp", pd.Series(dtype=object)).map(TOI_LABEL_MAP)
    frame = pd.DataFrame(
        {
            "name": renamed.get("toi").map(lambda t: f"TOI-{t}" if pd.notna(t) else None),
            "mission": "TESS",
            "star_id": renamed.get("kepid"),
            "ra": pd.to_numeric(renamed.get("ra"), errors="coerce"),
            "dec": pd.to_numeric(renamed.get("dec"), errors="coerce"),
            "dist_pc": pd.to_numeric(renamed.get("st_dist"), errors="coerce"),
            "disposition": label.map(DISPOSITION),
            "radius_earth": pd.to_numeric(renamed.get("koi_prad"), errors="coerce"),
            "period_days": pd.to_numeric(renamed.get("koi_period"), errors="coerce"),
        }
    )
    return frame, features


def build_skymap(
    koi: pd.DataFrame,
    stellar: pd.DataFrame,
    toi: pd.DataFrame,
    score: Scorer,
    reasons: Reasoner,
) -> pd.DataFrame:
    """Place both catalogs in space and attach the model's opinion of each.

    `score` and `reasons` are passed in rather than built here so the caller
    owns the model, and so this stays testable without one.
    """
    frames = []
    for frame, features in (_kepler_frame(koi, stellar), _tess_frame(toi)):
        aligned = features.loc[frame.index]
        out = frame.copy()
        out["probability"] = np.asarray(score(aligned), dtype=float)
        out["top_reasons"] = reasons(aligned)
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["ra", "dec", "dist_pc", "disposition"])
    combined = combined[combined["dist_pc"] > 0]
    combined["name"] = combined["name"].fillna("unnamed")
    return combined[SKYMAP_COLUMNS].reset_index(drop=True)
