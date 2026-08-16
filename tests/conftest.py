from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def koi_sample() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "koi_sample.csv", low_memory=False)


@pytest.fixture
def toi_sample() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "toi_sample.csv", low_memory=False)


@pytest.fixture
def koi_binary(koi_sample: pd.DataFrame):
    """Confirmed vs false positive, with features, labels, and star groups."""
    from exodiscover.features.tabular import build_features

    df = koi_sample[
        koi_sample["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
    ].reset_index(drop=True)
    y = (df["koi_disposition"] == "CONFIRMED").astype(int)
    return build_features(df), y, df["kepid"]
