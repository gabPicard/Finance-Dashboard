"""Tests for src.strategies.CAPM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.CAPM import CAPMStrategy, calculate_beta, capm_expected_returns


@pytest.fixture
def market_prices(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Synthetic market index prices derived from sample_prices mean."""
    mkt = sample_prices.mean(axis=1).to_frame("SPY")
    mkt.index.name = "date"
    return mkt


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_calculate_beta_is_finite(sample_prices: pd.DataFrame, market_prices: pd.DataFrame) -> None:
    """beta is a finite float for normal inputs."""
    asset_rets = sample_prices["AAPL"].pct_change().dropna()
    mkt_rets = market_prices["SPY"].pct_change().dropna()
    b = calculate_beta(asset_rets, mkt_rets)
    assert np.isfinite(b)


def test_capm_expected_returns_dict(sample_prices: pd.DataFrame, market_prices: pd.DataFrame) -> None:
    """capm_expected_returns returns a dict with all tickers as keys."""
    result = capm_expected_returns(sample_prices, market_prices, risk_free_rate=0.04)
    assert isinstance(result, dict)
    for ticker in sample_prices.columns:
        assert ticker in result
        assert np.isfinite(result[ticker])


def test_capm_strategy_run(sample_prices: pd.DataFrame, market_prices: pd.DataFrame) -> None:
    """CAPMStrategy.run() returns a Portfolio whose weights sum to 1."""
    from src.results.portfolio import Portfolio
    strategy = CAPMStrategy(
        name="CAPM_test",
        tickers=list(sample_prices.columns),
        start="2021-01-01",
        end="2022-12-31",
        prices=sample_prices.iloc[-252:],
        market_prices=market_prices.iloc[-252:],
        risk_free_rate=0.04,
    )
    portfolio = strategy.run()
    assert isinstance(portfolio, Portfolio)
    assert abs(sum(portfolio.weights.values()) - 1.0) < 1e-4


def test_capm_strategy_backtest_history(sample_prices: pd.DataFrame, market_prices: pd.DataFrame) -> None:
    """backtest() produces a non-empty weights_history."""
    strategy = CAPMStrategy(
        name="CAPM_test",
        tickers=list(sample_prices.columns),
        start="2021-01-01",
        end="2023-12-31",
        prices=sample_prices.iloc[-400:],
        market_prices=market_prices.iloc[-400:],
        window=120,
    )
    portfolio = strategy.backtest()
    assert not portfolio.weights_history.empty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_beta_against_itself_is_one(sample_prices: pd.DataFrame) -> None:
    """Beta of an asset against itself should be 1."""
    rets = sample_prices["AAPL"].pct_change().dropna()
    b = calculate_beta(rets, rets)
    assert b == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_capm_expected_returns_zero_variance_market() -> None:
    """When market has zero variance, beta fallback is 1.0."""
    asset = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0]})
    mkt = pd.DataFrame({"M": [100.0, 100.0, 100.0, 100.0]})
    result = capm_expected_returns(asset, mkt, risk_free_rate=0.04)
    assert "A" in result
    assert np.isfinite(result["A"])
