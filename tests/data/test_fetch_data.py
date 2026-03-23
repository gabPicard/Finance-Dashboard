"""Tests for src.data.fetch_data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cache import PriceCache
from src.data.fetch_data import DataUnavailableError, get_prices


# ---------------------------------------------------------------------------
# Happy path — cache hit
# ---------------------------------------------------------------------------

def test_get_prices_from_cache(paper_cache: PriceCache, sample_prices: pd.DataFrame) -> None:
    """get_prices returns data from the cache without hitting the DB or API."""
    import src.data.fetch_data as fd
    original_cache = fd.default_cache
    fd.default_cache = paper_cache
    try:
        result = get_prices("AAPL", "2021-01-01", "2023-01-01")
        assert isinstance(result, pd.DataFrame)
        assert "AAPL" in result.columns
    finally:
        fd.default_cache = original_cache


# ---------------------------------------------------------------------------
# Edge case — DB miss, API stub raises
# ---------------------------------------------------------------------------

def test_get_prices_raises_when_not_in_cache_or_db() -> None:
    """When data is absent from cache and DB, DataUnavailableError is raised."""
    import src.data.fetch_data as fd
    empty_cache = PriceCache(ttl=60)
    original_cache = fd.default_cache
    fd.default_cache = empty_cache
    try:
        with pytest.raises(DataUnavailableError):
            get_prices("FAKE_TICKER_XYZ", "2020-01-01", "2020-12-31")
    finally:
        fd.default_cache = original_cache


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_get_prices_empty_ticker_raises() -> None:
    """get_prices with empty string ticker raises DataUnavailableError."""
    with pytest.raises(DataUnavailableError):
        get_prices("", "2020-01-01", "2020-12-31")
