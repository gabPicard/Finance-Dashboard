"""Tests for src.results.portfolio — Portfolio dataclass."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.results.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_portfolio_validate_passes_when_weights_sum_to_one() -> None:
    """validate() does not raise when weights sum to 1.0."""
    portfolio = Portfolio(
        name="test",
        tickers=["AAPL", "MSFT"],
        weights={"AAPL": 0.6, "MSFT": 0.4},
    )
    portfolio.validate()  # should not raise


def test_portfolio_weight_series_indexed_by_ticker() -> None:
    """weight_series() returns a Series indexed by ticker."""
    portfolio = Portfolio(
        name="test",
        tickers=["AAPL", "MSFT"],
        weights={"AAPL": 0.6, "MSFT": 0.4},
    )
    s = portfolio.weight_series()
    assert s.index.tolist() == ["AAPL", "MSFT"]
    assert s["AAPL"] == pytest.approx(0.6)


def test_portfolio_repr_contains_name() -> None:
    """repr includes the portfolio name."""
    p = Portfolio(name="MyPortfolio", tickers=["A"], weights={"A": 1.0})
    assert "MyPortfolio" in repr(p)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_portfolio_validate_near_one_passes() -> None:
    """validate() passes for weights summing to 1.0 ± float tolerance."""
    w = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    portfolio = Portfolio(name="test", tickers=list(w.keys()), weights=w)
    portfolio.validate()


def test_portfolio_empty_weights_history_by_default() -> None:
    """weights_history is an empty DataFrame by default."""
    p = Portfolio(name="empty", tickers=[])
    assert p.weights_history.empty


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_portfolio_validate_raises_when_weights_not_summing_to_one() -> None:
    """validate() raises ValueError when weights sum != 1.0."""
    portfolio = Portfolio(
        name="bad",
        tickers=["A", "B"],
        weights={"A": 0.5, "B": 0.3},  # sum = 0.8
    )
    with pytest.raises(ValueError, match="weights sum"):
        portfolio.validate()


def test_portfolio_validate_raises_when_ticker_missing() -> None:
    """validate() raises ValueError when a ticker has no weight entry."""
    portfolio = Portfolio(
        name="missing",
        tickers=["AAPL", "MSFT", "GOOGL"],
        weights={"AAPL": 0.5, "MSFT": 0.5},  # GOOGL missing
    )
    with pytest.raises(ValueError, match="missing"):
        portfolio.validate()


def test_portfolio_validate_raises_when_no_weights() -> None:
    """validate() raises ValueError when weights dict is empty."""
    portfolio = Portfolio(name="empty_w", tickers=["AAPL"], weights={})
    with pytest.raises(ValueError):
        portfolio.validate()
