"""Query construction for the archive TAP service."""

import pytest

from exodiscover.data import ingest


def test_catalog_tables_are_fetched_whole():
    assert ingest.build_query("koi") == "select * from cumulative"


def test_the_stellar_table_is_fetched_by_column():
    """Q1_Q17_DR25_KS is 200,038 rows of 99 columns. Only four are wanted, and
    `select *` would pull roughly 200 MB to use 5.8 MB of it."""
    query = ingest.build_query("stellar")
    assert "select *" not in query
    for column in ("kepid", "dist", "dist_err1", "dist_err2"):
        assert column in query
    assert "Q1_Q17_DR25_KS" in query


def test_an_unknown_table_is_refused():
    with pytest.raises(KeyError, match="unknown table"):
        ingest.build_query("gaia")
