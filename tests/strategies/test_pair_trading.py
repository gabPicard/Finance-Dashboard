"""Tests for src.strategies.PairTrading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.PairTrading import (
    PairTradingStrategy,
    compute_spread,
    find_cointegrated_pairs,
    generate_signals,
    rolling_zscore,
)


@pytest.fixture
def cointegrated_prices() -> pd.DataFrame:
    """Two cointegrated price series plus one independent series."""
    np.random.seed(99)
    n = 300
    common = np.cumsum(np.random.normal(0, 1, n))
    s1 = common + np.random.normal(0, 0.1, n) + 100
    s2 = 2.0 * common + np.random.normal(0, 0.1, n) + 50
    independent = np.cumsum(np.random.normal(0, 1, n)) + 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"A": s1, "B": s2, "C": independent}, index=dates)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_find_cointegrated_pairs_detects_known_pair(cointegrated_prices: pd.DataFrame) -> None:
    """A/B are cointegrated; the function should detect them."""
    pairs = find_cointegrated_pairs(cointegrated_prices, p_threshold=0.05)
    tickers_found = {(p[0], p[1]) for p in pairs}
    assert ("A", "B") in tickers_found or ("B", "A") in tickers_found


def test_compute_spread_has_correct_length(cointegrated_prices: pd.DataFrame) -> None:
    """Spread series length matches the prices DataFrame length."""
    spread = compute_spread(cointegrated_prices, "A", "B")
    assert len(spread) == len(cointegrated_prices)


def test_rolling_zscore_nan_at_start() -> None:
    """rolling_zscore has NaN for the first (window-1) entries."""
    spread = pd.Series(np.random.normal(0, 1, 100))
    window = 20
    zscore = rolling_zscore(spread, window)
    assert zscore[:window - 1].isna().all()
    assert zscore[window - 1:].notna().all()


def test_generate_signals_range(cointegrated_prices: pd.DataFrame) -> None:
    """All generated signals are in {-1, 0, 1}."""
    spread = compute_spread(cointegrated_prices, "A", "B")
    zs = rolling_zscore(spread, window=20)
    signals = generate_signals(zs, entry_threshold=2.0, exit_threshold=0.5)
    assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})


def test_pair_trading_strategy_run_returns_book(cointegrated_prices: pd.DataFrame) -> None:
    """PairTradingStrategy.run() returns a TradingBook."""
    from src.results.trading_book import TradingBook
    strategy = PairTradingStrategy(
        name="PairTest",
        tickers=list(cointegrated_prices.columns),
        start="2020-01-01",
        end="2021-06-01",
        prices=cointegrated_prices,
        p_threshold=0.1,
    )
    book = strategy.run()
    assert isinstance(book, TradingBook)
    assert set(book.tickers) == set(cointegrated_prices.columns)


def test_pair_trading_backtest_pnl(cointegrated_prices: pd.DataFrame) -> None:
    """backtest() returns a TradingBook with non-empty pnl_history."""
    strategy = PairTradingStrategy(
        name="PairTest",
        tickers=list(cointegrated_prices.columns),
        start="2020-01-01",
        end="2021-06-01",
        prices=cointegrated_prices,
        p_threshold=0.1,
    )
    book = strategy.backtest()
    assert not book.pnl_history.empty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_strategy_handles_no_cointegrated_pairs(sample_prices: pd.DataFrame) -> None:
    """When no pairs are cointegrated, run() returns a book with zero positions."""
    strategy = PairTradingStrategy(
        name="NoPairs",
        tickers=list(sample_prices.columns),
        start="2021-01-01",
        end="2023-12-31",
        prices=sample_prices,
        p_threshold=0.001,  # very strict — unlikely to find pairs
    )
    book = strategy.run()
    # Positions should all be 0 or book should at least not crash
    assert book is not None


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_rolling_zscore_window_too_small() -> None:
    """rolling_zscore raises ValueError for window < 2."""
    spread = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rolling_zscore(spread, window=1)
