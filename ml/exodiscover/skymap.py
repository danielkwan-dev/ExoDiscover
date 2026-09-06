"""Builds the sky-map artifact: where each KOI actually is.

The KOI table carries `ra` and `dec` for every row but no distance, so a third
dimension has to come from the Kepler stellar table (`Q1_Q17_DR25_KS`), which
records a Gaia-derived `dist` in parsecs per star. Joining the two on `kepid`
places every object in real space with Earth at the origin.

Two facts about that data shape the result. The stellar table contains zeros
and nulls in the distance column, and neither is a distance -- a row that
cannot be placed is dropped rather than imputed, because substituting a median
would draw a star somewhere it is not. And the distances are large: the nearest
KOI sits around 250 pc and the median near 840 pc, because Kepler deliberately
observed a faint, distant field. Nothing here is a neighbour.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from exodiscover.features.tabular import add_multiplicity, build_features

#: Parsecs to light years.
LY_PER_PARSEC = 3.261563777

SKYMAP_COLUMNS: list[str] = [
    "kepoi_name",
    "kepler_name",
    "kepid",
    "ra",
    "dec",
    "dist_pc",
    "dist_err_pc",
    "disposition",
    "probability",
    "koi_prad",
    "koi_period",
]

#: Given a feature matrix, return one planet probability per row.
Scorer = Callable[[pd.DataFrame], np.ndarray]


def usable_distances(stellar: pd.DataFrame) -> pd.DataFrame:
    """One row per star, keeping only distances that mean something.

    The symmetric uncertainty is averaged from the archive's asymmetric pair;
    it is carried so the UI can show a range rather than implying that a
    distance measured to within about 19% is exact.
    """
    out = stellar.dropna(subset=["dist"])
    out = out[out["dist"] > 0].drop_duplicates("kepid")

    err = pd.Series(np.nan, index=out.index, dtype=float)
    if {"dist_err1", "dist_err2"} <= set(out.columns):
        err = (out["dist_err1"].abs() + out["dist_err2"].abs()) / 2.0

    return pd.DataFrame({"kepid": out["kepid"], "dist_pc": out["dist"], "dist_err_pc": err})


def build_skymap(koi: pd.DataFrame, stellar: pd.DataFrame, score: Scorer) -> pd.DataFrame:
    """Join positions to distances and score every object in the catalog.

    Every disposition is kept, including the unvetted candidates -- those are
    the rows the model has an opinion about that nobody has confirmed yet, and
    they are the point of showing the map at all.
    """
    counted = add_multiplicity(koi)
    probability = np.asarray(score(build_features(counted)), dtype=float)

    frame = pd.DataFrame(
        {
            "kepoi_name": counted.get("kepoi_name"),
            "kepler_name": counted.get("kepler_name"),
            "kepid": counted["kepid"],
            "ra": pd.to_numeric(counted.get("ra"), errors="coerce"),
            "dec": pd.to_numeric(counted.get("dec"), errors="coerce"),
            "disposition": counted.get("koi_disposition"),
            "probability": probability,
            "koi_prad": pd.to_numeric(counted.get("koi_prad"), errors="coerce"),
            "koi_period": pd.to_numeric(counted.get("koi_period"), errors="coerce"),
        }
    )

    merged = frame.merge(usable_distances(stellar), on="kepid", how="inner")
    merged = merged.dropna(subset=["ra", "dec", "dist_pc"])
    return merged[SKYMAP_COLUMNS].reset_index(drop=True)
