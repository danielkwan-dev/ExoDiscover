import pandas as pd
import pytest

from exodiscover.data.schema import (
    KOI_LABEL_MAP,
    LEAKY_COLUMNS,
    TOI_LABEL_MAP,
    TOI_RESOLVED,
    LeakageError,
    assert_no_leakage,
)


def test_known_leaky_columns_are_listed():
    for col in [
        "koi_score",
        "koi_pdisposition",
        "koi_disposition",
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
        "koi_tce_delivname",
    ]:
        assert col in LEAKY_COLUMNS


def test_assert_no_leakage_passes_on_clean_frame():
    assert_no_leakage(pd.DataFrame({"koi_period": [1.0], "koi_depth": [100.0]}))


def test_assert_no_leakage_raises_and_names_every_offender():
    df = pd.DataFrame({"koi_period": [1.0], "koi_score": [0.9], "koi_fpflag_ss": [0]})
    with pytest.raises(LeakageError) as exc:
        assert_no_leakage(df, context="feature matrix")
    message = str(exc.value)
    assert "koi_score" in message
    assert "koi_fpflag_ss" in message
    assert "feature matrix" in message


def test_toi_known_planet_maps_to_confirmed():
    # Regression: the original preprocess_merge.py mapped KP to Candidate.
    assert TOI_LABEL_MAP["KP"] == 2
    assert TOI_LABEL_MAP["CP"] == 2


def test_toi_false_alarm_is_not_dropped():
    # Regression: the original map omitted APC and FA, silently producing NaN.
    assert TOI_LABEL_MAP["FA"] == 0
    assert TOI_LABEL_MAP["APC"] == 1


def test_unvetted_toi_dispositions_are_not_treated_as_resolved():
    assert "PC" not in TOI_RESOLVED
    assert "APC" not in TOI_RESOLVED
    assert set(TOI_RESOLVED) == {"CP", "KP", "FP", "FA"}


def test_koi_label_map_is_ordinal():
    assert KOI_LABEL_MAP["FALSE POSITIVE"] < KOI_LABEL_MAP["CANDIDATE"]
    assert KOI_LABEL_MAP["CANDIDATE"] < KOI_LABEL_MAP["CONFIRMED"]
