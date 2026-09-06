"""The sky-map artifact: real positions, real distances, nothing invented."""

import numpy as np
import pandas as pd
import pytest

from exodiscover import skymap


@pytest.fixture
def stellar(koi_sample: pd.DataFrame) -> pd.DataFrame:
    """A stand-in for Q1_Q17_DR25_KS: one distance per star, some unusable."""
    kepids = koi_sample["kepid"].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(0)
    dist = pd.Series(rng.uniform(250.0, 1900.0, size=len(kepids)))
    # The real table contains both of these, and neither is a distance.
    dist.iloc[0] = 0.0
    dist.iloc[1] = np.nan
    return pd.DataFrame(
        {
            "kepid": kepids,
            "dist": dist,
            "dist_err1": dist * 0.19,
            "dist_err2": -dist * 0.19,
        }
    )


@pytest.fixture
def scored(koi_sample: pd.DataFrame, stellar: pd.DataFrame) -> pd.DataFrame:
    # A stand-in scorer: the artifact's shape must not depend on the model.
    return skymap.build_skymap(koi_sample, stellar, lambda X: np.full(len(X), 0.5))


def test_columns_are_exactly_the_contract(scored):
    assert list(scored.columns) == skymap.SKYMAP_COLUMNS


def test_every_row_can_actually_be_placed(scored):
    """A point with no position or no distance cannot be drawn, so it must not
    reach the artifact at all rather than becoming a NaN the UI has to guess at."""
    assert scored["ra"].notna().all()
    assert scored["dec"].notna().all()
    assert scored["dist_pc"].notna().all()
    assert (scored["dist_pc"] > 0).all()


def test_unusable_distances_are_dropped_not_repaired(koi_sample, stellar):
    """The stellar table carries zeros and nulls. Neither is a distance, and
    substituting a median would place a star somewhere it is not."""
    out = skymap.build_skymap(koi_sample, stellar, lambda X: np.full(len(X), 0.5))
    bad_stars = set(stellar.loc[stellar["dist"].isna() | (stellar["dist"] <= 0), "kepid"])
    assert not (set(out["kepid"]) & bad_stars)
    assert len(out) < len(koi_sample), "the fixture's bad rows should have been removed"


def test_distance_is_joined_from_the_stellar_table(koi_sample, stellar):
    out = skymap.build_skymap(koi_sample, stellar, lambda X: np.full(len(X), 0.5))
    usable = stellar[stellar["dist"].notna() & (stellar["dist"] > 0)]
    expected = dict(zip(usable["kepid"], usable["dist"], strict=True))
    for _, row in out.head(20).iterrows():
        assert row["dist_pc"] == pytest.approx(expected[row["kepid"]])


def test_probability_comes_from_the_scorer(koi_sample, stellar):
    out = skymap.build_skymap(koi_sample, stellar, lambda X: np.full(len(X), 0.25))
    assert (out["probability"] == 0.25).all()
    assert out["probability"].between(0, 1).all()


def test_every_disposition_survives(scored, koi_sample):
    """The map shows the whole catalog, not just the resolved part - the
    unvetted candidates are the interesting ones."""
    assert set(scored["disposition"]) <= set(koi_sample["koi_disposition"].dropna())
    assert len(set(scored["disposition"])) > 1
