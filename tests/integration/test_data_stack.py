"""Integration test: data stack — cache → DB → API stub chain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cache import PriceCache
from src.data.fetch_data import DataUnavailableError, get_prices


# ---------------------------------------------------------------------------
# Happy path — cache layer
# ---------------------------------------------------------------------------

def test_cache_hit_returns_data(sample_prices: pd.DataFrame) -> None:
    """A warm cache returns data without touching the DB."""
    import src.data.fetch_data as fd

    cache = PriceCache(ttl=3600)
    df = sample_prices[["AAPL"]]
    cache.set("AAPL", "2021-01-01", "2023-01-01", df)

    original = fd.default_cache
    fd.default_cache = cache
    try:
        result = get_prices("AAPL", "2021-01-01", "2023-01-01")
        assert result is not None
        assert "AAPL" in result.columns
        assert len(result) > 0
    finally:
        fd.default_cache = original


# ---------------------------------------------------------------------------
# Happy path — multiple tickers cached independently
# ---------------------------------------------------------------------------

def test_each_ticker_cached_independently(sample_prices: pd.DataFrame) -> None:
    """Separate cache entries exist for each ticker."""
    import src.data.fetch_data as fd

    cache = PriceCache(ttl=3600)
    for ticker in sample_prices.columns:
        cache.set(ticker, "2021-01-01", "2023-01-01", sample_prices[[ticker]])

    original = fd.default_cache
    fd.default_cache = cache
    try:
        for ticker in sample_prices.columns:
            result = get_prices(ticker, "2021-01-01", "2023-01-01")
            assert ticker in result.columns
    finally:
        fd.default_cache = original


# ---------------------------------------------------------------------------
# Edge case — API stub raises for unknown tickers
# ---------------------------------------------------------------------------

def test_api_stub_raises_for_unknown_ticker() -> None:
    """The API stub raises DataUnavailableError for any unknown ticker."""
    import src.data.fetch_data as fd

    empty_cache = PriceCache(ttl=60)
    original = fd.default_cache
    fd.default_cache = empty_cache
    try:
        with pytest.raises(DataUnavailableError):
            get_prices("DOES_NOT_EXIST_XYZ", "2023-01-01", "2023-12-31")
    finally:
        fd.default_cache = original


# ---------------------------------------------------------------------------
# Integration: cache invalidation reflects updated data
# ---------------------------------------------------------------------------

def test_cache_invalidation_then_miss(sample_prices: pd.DataFrame) -> None:
    """After invalidation, get_prices raises (no DB/API for stub tickers)."""
    import src.data.fetch_data as fd

    cache = PriceCache(ttl=3600)
    cache.set("AAPL", "2021-01-01", "2023-01-01", sample_prices[["AAPL"]])
    cache.invalidate("AAPL", "2021-01-01", "2023-01-01")

    original = fd.default_cache
    fd.default_cache = cache
    try:
        with pytest.raises(DataUnavailableError):
            get_prices("AAPL", "2021-01-01", "2023-01-01")
    finally:
        fd.default_cache = original
