import pandas as pd
import pytest

from exodiscover.data.schema import (
    KOI_LABEL_MAP,
    LEAKY_COLUMNS,
    MAX_LEAK_CORRELATION,
    TOI_LABEL_MAP,
    TOI_RESOLVED,
    LeakageError,
    assert_no_derived_leakage,
    assert_no_leakage,
    find_derived_leakage,
)
from exodiscover.features.tabular import build_features


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


def test_name_check_cannot_see_a_renamed_leaky_column():
    """States the limit the value check exists to cover.

    assert_no_leakage compares column names, so a Robovetter column shipped
    under any other name passes it untouched.
    """
    df = pd.DataFrame({"koi_period": [1.0, 2.0], "vetting_confidence": [0.9, 0.1]})
    assert_no_leakage(df)  # passes, and should not


def test_derived_leakage_is_caught_when_a_leaky_column_is_renamed(koi_sample):
    X = build_features(koi_sample)
    X["vetting_confidence"] = pd.to_numeric(koi_sample["koi_score"], errors="coerce")

    with pytest.raises(LeakageError) as exc:
        assert_no_derived_leakage(X, koi_sample, context="feature matrix")

    message = str(exc.value)
    assert "vetting_confidence" in message
    assert "koi_score" in message
    assert "feature matrix" in message


def test_derived_leakage_survives_dilution(koi_sample):
    """A leak does not have to be a clean copy to be a leak."""
    X = build_features(koi_sample)
    score = pd.to_numeric(koi_sample["koi_score"], errors="coerce")
    X["diluted"] = score * 0.5 + X["log_prad"] * 0.01

    offenders = find_derived_leakage(X, koi_sample)
    assert any(feature == "diluted" for feature, _, _ in offenders)


def test_the_shipped_feature_matrix_has_no_derived_leakage(koi_sample):
    """The real guard: every feature that actually ships must pass."""
    offenders = find_derived_leakage(build_features(koi_sample), koi_sample)
    assert offenders == [], f"above {MAX_LEAK_CORRELATION}: {offenders}"


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
