"""Tests for src.results.trading_book — TradingBook dataclass."""

from __future__ import annotations

import pytest

from src.results.trading_book import TradingBook


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_net_exposure_sum_of_absolutes() -> None:
    """net_exposure() returns the sum of absolute position values."""
    book = TradingBook(
        name="test",
        tickers=["A", "B"],
        positions={"A": 1.0, "B": -0.5},
    )
    assert book.net_exposure() == pytest.approx(1.5)


def test_net_position_is_algebraic_sum() -> None:
    """net_position() returns the signed sum."""
    book = TradingBook(
        name="test",
        tickers=["A", "B"],
        positions={"A": 1.0, "B": -1.0},
    )
    assert book.net_position() == pytest.approx(0.0)


def test_position_series_indexed_by_ticker() -> None:
    """position_series() is indexed by ticker."""
    book = TradingBook(
        name="test",
        tickers=["X", "Y"],
        positions={"X": 2.0, "Y": -2.0},
    )
    s = book.position_series()
    assert s["X"] == pytest.approx(2.0)
    assert s["Y"] == pytest.approx(-2.0)


def test_trading_book_repr_contains_name() -> None:
    """repr contains the book name."""
    book = TradingBook(name="PairBook", tickers=["A"], positions={"A": 1.0})
    assert "PairBook" in repr(book)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_net_exposure_with_no_positions() -> None:
    """net_exposure() returns 0 when positions dict is empty."""
    book = TradingBook(name="empty", tickers=[])
    assert book.net_exposure() == 0.0


def test_net_position_with_all_long() -> None:
    """net_position() equals net_exposure for all-long positions."""
    book = TradingBook(
        name="long",
        tickers=["A", "B"],
        positions={"A": 0.5, "B": 0.5},
    )
    assert book.net_position() == book.net_exposure()


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_trading_book_empty_pnl_history_by_default() -> None:
    """pnl_history is an empty Series by default."""
    book = TradingBook(name="no_pnl", tickers=[])
    assert book.pnl_history.empty
