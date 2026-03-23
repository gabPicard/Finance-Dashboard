"""Shared pytest fixtures for the Finance Dashboard test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.cache import PriceCache
from src.db.models import Base


# ---------------------------------------------------------------------------
# Price fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_prices() -> pd.DataFrame:
    """500 days of synthetic close prices for 3 tickers (AAPL, MSFT, GOOGL).

    Returns a DataFrame with DatetimeIndex and tickers as columns.
    """
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2021-01-01", periods=n)
    prices = {}
    for ticker, start_price in [("AAPL", 150.0), ("MSFT", 300.0), ("GOOGL", 2800.0)]:
        log_rets = np.random.normal(0.0005, 0.015, n)
        price_series = start_price * np.exp(np.cumsum(log_rets))
        prices[ticker] = price_series
    df = pd.DataFrame(prices, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture(scope="session")
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns of sample_prices."""
    return np.log(sample_prices / sample_prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def in_memory_db():
    """SQLAlchemy session connected to an in-memory SQLite database.

    All tables defined in the ORM models are created before yielding.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Paper-mode handler fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def paper_cache(sample_prices: pd.DataFrame) -> PriceCache:
    """A PriceCache pre-loaded with sample_prices for all 3 tickers.

    The cache key for each ticker is ``"{ticker}:2021-01-01:2023-01-01"``.
    """
    cache = PriceCache(ttl=3600)
    for ticker in sample_prices.columns:
        df = sample_prices[[ticker]]
        cache.set(ticker, "2021-01-01", "2023-01-01", df)
    return cache
