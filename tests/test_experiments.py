import numpy as np
import pandas as pd

from exodiscover.experiments import ablation, transfer
from exodiscover.features.tabular import FEATURE_COLUMNS


def test_collapse_maps_confirmed_and_candidate_to_planetlike():
    out = ablation.collapse_to_planetlike(np.array([0, 1, 2, 0, 2]))
    assert out.tolist() == [0, 1, 1, 0, 1]


def test_leakage_ablation_covers_all_three_setups(koi_sample):
    rows = ablation.run_leakage_ablation(koi_sample)
    assert {r["setup"] for r in rows} == {
        "leaky_random_split",
        "clean_random_split",
        "clean_grouped_split",
    }
    for row in rows:
        assert 0.0 <= row["roc_auc"] <= 1.0
        assert row["note"]


def test_leaky_setup_scores_higher_than_the_honest_one(koi_sample):
    """If this fails, the firewall is misconfigured: adding the Robovetter's
    own verdict columns must make the score go up."""
    by_setup = {r["setup"]: r for r in ablation.run_leakage_ablation(koi_sample)}
    assert by_setup["leaky_random_split"]["roc_auc"] > by_setup["clean_grouped_split"]["roc_auc"]


def test_framing_ablation_reports_all_three_framings(koi_sample):
    rows = ablation.run_framing_ablation(koi_sample)
    assert {r["framing"] for r in rows} == {"3class", "binary", "diagnostic"}
    for row in rows:
        assert 0.0 <= row["pr_auc"] <= 1.0
        assert row["target"]


def test_shared_features_are_a_subset_of_the_kepler_features():
    assert set(transfer.SHARED_FEATURES) <= set(FEATURE_COLUMNS)
    assert len(transfer.SHARED_FEATURES) >= 8


def test_ambiguous_toi_dispositions_are_excluded():
    toi = pd.DataFrame(
        {
            "tid": [1, 2, 3, 4, 5, 6],
            "tfopwg_disp": ["CP", "KP", "FP", "FA", "PC", "APC"],
            "pl_orbper": [3.0] * 6,
            "pl_trandep": [500.0] * 6,
            "pl_trandurh": [2.0] * 6,
            "pl_rade": [2.0] * 6,
            "st_teff": [5500.0] * 6,
            "st_logg": [4.4] * 6,
            "st_rad": [1.0] * 6,
            "pl_insol": [10.0] * 6,
            "pl_eqt": [600.0] * 6,
            "st_tmag": [11.0] * 6,
        }
    )
    X, y = transfer.toi_to_common(toi)
    assert len(X) == 4, "PC and APC are unvetted and must not become labels"
    assert y.tolist() == [1, 1, 0, 0], "KP is a confirmed planet, not a candidate"


def test_transfer_reports_both_domains(koi_sample, toi_sample):
    result = transfer.run_transfer(koi_sample, toi_sample)
    assert 0.0 <= result["zero_shot"]["roc_auc"] <= 1.0
    assert 0.0 <= result["in_domain"]["roc_auc"] <= 1.0
    assert result["zero_shot"]["n"] > 0
    assert set(result["in_domain"]) >= {"n", "base_rate", "roc_auc", "pr_auc", "brier"}
