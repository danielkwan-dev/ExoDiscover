"""The sky-map artifact: both missions, real positions, nothing invented."""

import numpy as np
import pandas as pd
import pytest

from exodiscover import skymap


def _scorer(p: float = 0.5):
    return lambda X: np.full(len(X), p)


def _reasoner(X: pd.DataFrame) -> list[str]:
    return ["log_prad:+1.00;duration_ratio:-0.50"] * len(X)


@pytest.fixture
def stellar(koi_sample: pd.DataFrame) -> pd.DataFrame:
    """A stand-in for Q1_Q17_DR25_KS: one distance per star, some unusable."""
    kepids = koi_sample["kepid"].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(0)
    dist = pd.Series(rng.uniform(250.0, 1900.0, size=len(kepids)))
    # The real table contains both of these, and neither is a distance.
    dist.iloc[0] = 0.0
    dist.iloc[1] = np.nan
    return pd.DataFrame({"kepid": kepids, "dist": dist, "dist_err1": dist * 0.19})


@pytest.fixture
def built(koi_sample, stellar, toi_sample):
    return skymap.build_skymap(koi_sample, stellar, toi_sample, _scorer(), _reasoner)


def test_columns_are_exactly_the_contract(built):
    assert list(built.columns) == skymap.SKYMAP_COLUMNS


def test_both_missions_are_present(built):
    """TESS is what makes the view surround you: Kepler stared at one 22x16
    degree patch, TESS covered the whole sky."""
    assert set(built["mission"]) == {"Kepler", "TESS"}
    assert (built["mission"] == "Kepler").sum() > 0
    assert (built["mission"] == "TESS").sum() > 0


def test_every_row_can_actually_be_placed(built):
    for column in ("ra", "dec", "dist_pc"):
        assert built[column].notna().all()
    assert (built["dist_pc"] > 0).all()


def test_tess_covers_sky_kepler_does_not(built):
    """A sanity check on the thing the map exists to show."""
    tess = built[built["mission"] == "TESS"]
    kepler = built[built["mission"] == "Kepler"]
    assert tess["dec"].min() < kepler["dec"].min()
    assert tess["ra"].max() - tess["ra"].min() > kepler["ra"].max() - kepler["ra"].min()


def test_tess_dispositions_are_mapped_to_the_shared_vocabulary(built):
    """The two archives disagree on names. CP and KP are both confirmed
    planets, PC and APC are unvetted, FP and FA are neither."""
    assert set(built["disposition"]) <= {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}
    tess = built[built["mission"] == "TESS"]
    assert len(set(tess["disposition"])) > 1


def test_unusable_distances_are_dropped_not_repaired(koi_sample, stellar, toi_sample):
    out = skymap.build_skymap(koi_sample, stellar, toi_sample, _scorer(), _reasoner)
    bad = set(stellar.loc[stellar["dist"].isna() | (stellar["dist"] <= 0), "kepid"])
    kepler_ids = set(out.loc[out["mission"] == "Kepler", "star_id"])
    assert not (kepler_ids & bad)


def test_probability_comes_from_one_model_for_both_missions(koi_sample, stellar, toi_sample):
    """Scoring the two catalogs with different models would make the numbers
    incomparable; only the 11 shared features can serve both."""
    out = skymap.build_skymap(koi_sample, stellar, toi_sample, _scorer(0.25), _reasoner)
    assert (out["probability"] == 0.25).all()


def test_reasons_are_precomputed_and_parseable(built):
    """The panel reads these directly, so no scoring round-trip on click."""
    assert built["top_reasons"].str.len().gt(0).all()
    first = built["top_reasons"].iloc[0]
    for part in first.split(";"):
        name, value = part.rsplit(":", 1)
        assert name
        float(value)
