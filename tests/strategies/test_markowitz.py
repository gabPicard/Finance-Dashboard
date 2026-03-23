"""Tests for src.strategies.Markowitz."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.Markowitz import (
    MarkowitzStrategy,
    max_sharpe_portfolio,
    optimize_portfolio,
    portfolio_performance,
)


@pytest.fixture
def small_prices(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Use 2 years of the first 3 tickers."""
    return sample_prices.iloc[-252:].copy()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_min_var_weights_sum_to_one(small_prices: pd.DataFrame) -> None:
    """Minimum-variance weights sum to 1.0."""
    rets = small_prices.pct_change().dropna()
    cov = rets.cov() * 252
    mu = rets.mean() * 252
    w = optimize_portfolio(cov, mu)
    assert w is not None
    assert abs(w.sum() - 1.0) < 1e-4


def test_max_sharpe_weights_positive(small_prices: pd.DataFrame) -> None:
    """All max-Sharpe weights are >= 0 (long-only)."""
    rets = small_prices.pct_change().dropna()
    cov = rets.cov() * 252
    mu = rets.mean() * 252
    w = max_sharpe_portfolio(cov, mu, risk_free_rate=0.04, max_weight=1.0)
    assert w is not None
    assert (w >= -1e-8).all()


def test_max_sharpe_weight_cap_respected(small_prices: pd.DataFrame) -> None:
    """No single weight exceeds max_weight."""
    rets = small_prices.pct_change().dropna()
    cov = rets.cov() * 252
    mu = rets.mean() * 252
    cap = 0.5
    w = max_sharpe_portfolio(cov, mu, max_weight=cap)
    assert w is not None
    assert (w <= cap + 1e-4).all()


def test_markowitz_strategy_run_returns_portfolio(small_prices: pd.DataFrame) -> None:
    """MarkowitzStrategy.run() returns a Portfolio with correct tickers."""
    from src.results.portfolio import Portfolio
    strategy = MarkowitzStrategy(
        name="test", tickers=list(small_prices.columns),
        start="2022-01-01", end="2023-12-31",
        prices=small_prices
    )
    portfolio = strategy.run()
    assert isinstance(portfolio, Portfolio)
    assert abs(sum(portfolio.weights.values()) - 1.0) < 1e-4


def test_markowitz_strategy_backtest_weights_history(small_prices: pd.DataFrame) -> None:
    """backtest() populates weights_history."""
    strategy = MarkowitzStrategy(
        name="test", tickers=list(small_prices.columns),
        start="2021-01-01", end="2023-12-31",
        prices=small_prices, window=120,
    )
    portfolio = strategy.backtest()
    assert not portfolio.weights_history.empty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_portfolio_performance_non_negative_std() -> None:
    """portfolio_performance std is never negative."""
    w = np.array([0.5, 0.5])
    mu = np.array([0.1, 0.05])
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    perf = portfolio_performance(w, mu, cov)
    assert perf["std"] >= 0


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_optimize_portfolio_single_asset() -> None:
    """optimize_portfolio with a single asset returns [1.0]."""
    cov = np.array([[0.04]])
    mu = np.array([0.1])
    w = optimize_portfolio(cov, mu)
    assert w is not None
    assert w[0] == pytest.approx(1.0, abs=1e-4)
