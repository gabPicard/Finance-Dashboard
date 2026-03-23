"""Tests for src.metrics.portfolio_measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.portfolio_measurements import (
    alpha,
    annualised_return,
    annualised_volatility,
    beta,
    max_drawdown,
    sharpe_ratio,
    value_at_risk,
)


@pytest.fixture
def flat_returns() -> pd.Series:
    """Returns of 0 every day."""
    return pd.Series(np.zeros(252))


@pytest.fixture
def positive_returns() -> pd.Series:
    """Constant small positive daily returns (~12.7% annual)."""
    return pd.Series(np.full(252, 0.0005))


@pytest.fixture
def volatile_returns(sample_returns: pd.DataFrame) -> pd.Series:
    """Use the first ticker column from sample_returns."""
    return sample_returns.iloc[:, 0]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_sharpe_ratio_positive(positive_returns: pd.Series) -> None:
    """Sharpe ratio is positive for consistently positive returns."""
    sr = sharpe_ratio(positive_returns, risk_free_rate=0.0)
    assert sr > 0


def test_max_drawdown_negative(volatile_returns: pd.Series) -> None:
    """max_drawdown returns a non-positive value."""
    mdd = max_drawdown(volatile_returns)
    assert mdd <= 0


def test_annualised_return_positive(positive_returns: pd.Series) -> None:
    """annualised_return is positive for positive returns."""
    ar = annualised_return(positive_returns)
    assert ar > 0


def test_annualised_volatility_zero_for_flat(flat_returns: pd.Series) -> None:
    """Volatility is 0 for constant returns."""
    vol = annualised_volatility(flat_returns)
    assert vol == pytest.approx(0.0, abs=1e-10)


def test_var_is_negative(volatile_returns: pd.Series) -> None:
    """VaR at 95% is <= 0 (loss)."""
    var = value_at_risk(volatile_returns, confidence=0.95)
    assert var <= 0


def test_beta_one_for_identical_series(volatile_returns: pd.Series) -> None:
    """Beta of an asset against itself is 1."""
    b = beta(volatile_returns, volatile_returns)
    assert b == pytest.approx(1.0, abs=1e-6)


def test_alpha_zero_for_market_portfolio(volatile_returns: pd.Series) -> None:
    """Alpha of the market portfolio against itself is ~0."""
    a = alpha(volatile_returns, volatile_returns, risk_free_rate=0.0)
    assert abs(a) < 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_sharpe_zero_for_flat_returns(flat_returns: pd.Series) -> None:
    """Sharpe ratio is 0 when std is 0."""
    sr = sharpe_ratio(flat_returns)
    assert sr == 0.0


def test_max_drawdown_empty_returns() -> None:
    """max_drawdown returns 0 for empty Series."""
    mdd = max_drawdown(pd.Series(dtype=float))
    assert mdd == 0.0


def test_annualised_return_empty() -> None:
    """annualised_return returns 0 for empty Series."""
    ar = annualised_return(pd.Series(dtype=float))
    assert ar == 0.0


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_var_confidence_zero() -> None:
    """value_at_risk with confidence=0 returns max loss (worst return)."""
    rets = pd.Series([-0.05, -0.03, 0.01, 0.02, 0.03])
    var = value_at_risk(rets, confidence=0.0)
    assert var <= 0
