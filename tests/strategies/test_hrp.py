"""Tests for src.strategies.HRP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.HRP import HRPStrategy, hrp_weights


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_hrp_weights_sum_to_one(sample_returns: pd.DataFrame) -> None:
    """hrp_weights returns weights that sum to 1."""
    w = hrp_weights(sample_returns)
    assert abs(w.sum() - 1.0) < 1e-6


def test_hrp_weights_all_positive(sample_returns: pd.DataFrame) -> None:
    """HRP produces non-negative weights (long-only)."""
    w = hrp_weights(sample_returns)
    assert (w >= 0).all()


def test_hrp_weights_cover_all_tickers(sample_returns: pd.DataFrame) -> None:
    """hrp_weights returns one weight per ticker."""
    w = hrp_weights(sample_returns)
    assert set(w.index) == set(sample_returns.columns)


def test_hrp_strategy_run(sample_prices: pd.DataFrame) -> None:
    """HRPStrategy.run() returns a Portfolio with weights summing to 1."""
    from src.results.portfolio import Portfolio
    strategy = HRPStrategy(
        name="HRP_test",
        tickers=list(sample_prices.columns),
        start="2021-01-01",
        end="2023-12-31",
        prices=sample_prices.iloc[-252:],
    )
    portfolio = strategy.run()
    assert isinstance(portfolio, Portfolio)
    assert abs(sum(portfolio.weights.values()) - 1.0) < 1e-4


def test_hrp_strategy_backtest_history(sample_prices: pd.DataFrame) -> None:
    """backtest() populates weights_history."""
    strategy = HRPStrategy(
        name="HRP_test",
        tickers=list(sample_prices.columns),
        start="2021-01-01",
        end="2023-12-31",
        prices=sample_prices.iloc[-400:],
        window=120,
    )
    portfolio = strategy.backtest()
    assert not portfolio.weights_history.empty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_hrp_equal_weight_when_uncorrelated() -> None:
    """Three uncorrelated assets with equal variance receive equal HRP weight."""
    np.random.seed(0)
    n = 300
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.01, n),
        "B": np.random.normal(0, 0.01, n),
        "C": np.random.normal(0, 0.01, n),
    })
    w = hrp_weights(returns)
    # All weights should be close to 1/3
    for ticker in ["A", "B", "C"]:
        assert abs(w[ticker] - 1/3) < 0.1


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_hrp_single_asset() -> None:
    """hrp_weights with a single asset assigns weight 1.0."""
    returns = pd.DataFrame({"SOLO": np.random.normal(0, 0.01, 100)})
    w = hrp_weights(returns)
    assert w["SOLO"] == pytest.approx(1.0, abs=1e-6)
