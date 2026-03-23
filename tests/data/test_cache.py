"""Tests for src.data.cache — PriceCache."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.data.cache import PriceCache


@pytest.fixture
def cache() -> PriceCache:
    return PriceCache(ttl=5)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=10, freq="B")
    return pd.DataFrame({"AAPL": np.random.rand(10)}, index=idx)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_set_and_get(cache: PriceCache, sample_df: pd.DataFrame) -> None:
    """A value stored with set() is retrievable with get()."""
    cache.set("AAPL", "2022-01-01", "2022-01-14", sample_df)
    result = cache.get("AAPL", "2022-01-01", "2022-01-14")
    assert result is not None
    pd.testing.assert_frame_equal(result, sample_df)


def test_len_increments(cache: PriceCache, sample_df: pd.DataFrame) -> None:
    """len() reflects the number of stored entries."""
    assert len(cache) == 0
    cache.set("AAPL", "2022-01-01", "2022-01-14", sample_df)
    assert len(cache) == 1
    cache.set("MSFT", "2022-01-01", "2022-01-14", sample_df)
    assert len(cache) == 2


def test_clear(cache: PriceCache, sample_df: pd.DataFrame) -> None:
    """clear() removes all entries."""
    cache.set("AAPL", "2022-01-01", "2022-01-14", sample_df)
    cache.clear()
    assert len(cache) == 0
    assert cache.get("AAPL", "2022-01-01", "2022-01-14") is None


def test_invalidate_specific_entry(cache: PriceCache, sample_df: pd.DataFrame) -> None:
    """invalidate() removes only the targeted entry."""
    cache.set("AAPL", "2022-01-01", "2022-01-14", sample_df)
    cache.set("MSFT", "2022-01-01", "2022-01-14", sample_df)
    cache.invalidate("AAPL", "2022-01-01", "2022-01-14")
    assert cache.get("AAPL", "2022-01-01", "2022-01-14") is None
    assert cache.get("MSFT", "2022-01-01", "2022-01-14") is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_get_missing_key(cache: PriceCache) -> None:
    """Retrieving a non-existent key returns None."""
    assert cache.get("TSLA", "2022-01-01", "2022-12-31") is None


def test_ttl_expiry(sample_df: pd.DataFrame) -> None:
    """Entries expire after TTL elapses."""
    short_cache = PriceCache(ttl=1)
    short_cache.set("AAPL", "2022-01-01", "2022-01-14", sample_df)
    time.sleep(1.1)
    assert short_cache.get("AAPL", "2022-01-01", "2022-01-14") is None


def test_evict_expired(sample_df: pd.DataFrame) -> None:
    """evict_expired() removes stale entries and returns count."""
    short_cache = PriceCache(ttl=1)
    short_cache.set("AAPL", "2022-01-01", "2022-01-14", sample_df)
    time.sleep(1.1)
    evicted = short_cache.evict_expired()
    assert evicted == 1
    assert len(short_cache) == 0


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_make_key_format() -> None:
    """make_key produces the expected format."""
    key = PriceCache.make_key("AAPL", "2022-01-01", "2022-12-31")
    assert key == "AAPL:2022-01-01:2022-12-31"
