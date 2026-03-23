"""Tests for src.db.models — ORM model integrity."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from src.db.models import BacktestRun, Corporation, Market, PriceHistory


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_create_market(in_memory_db: Session) -> None:
    """A Market row can be inserted and queried back."""
    market = Market(name="S&P 500", exchange="NYSE", region="US", currency="USD")
    in_memory_db.add(market)
    in_memory_db.commit()

    result = in_memory_db.query(Market).filter_by(name="S&P 500").first()
    assert result is not None
    assert result.exchange == "NYSE"


def test_create_corporation_with_market_fk(in_memory_db: Session) -> None:
    """A Corporation with a valid market_id FK can be inserted."""
    market = Market(name="NASDAQ", exchange="NASDAQ", region="US", currency="USD")
    in_memory_db.add(market)
    in_memory_db.flush()

    corp = Corporation(
        ticker="AAPL",
        name="Apple Inc.",
        sector="Technology",
        market_id=market.id,
    )
    in_memory_db.add(corp)
    in_memory_db.commit()

    result = in_memory_db.query(Corporation).filter_by(ticker="AAPL").first()
    assert result is not None
    assert result.name == "Apple Inc."
    assert result.market_id == market.id


def test_price_history_unique_constraint(in_memory_db: Session) -> None:
    """Inserting a duplicate (ticker, date) raises an IntegrityError."""
    from sqlalchemy.exc import IntegrityError

    market = Market(name="Test", exchange="Test", region="US", currency="USD")
    in_memory_db.add(market)
    in_memory_db.flush()
    corp = Corporation(ticker="TEST", name="Test Corp", market_id=market.id)
    in_memory_db.add(corp)
    in_memory_db.flush()

    row1 = PriceHistory(ticker="TEST", date="2022-01-01", close=100.0)
    in_memory_db.add(row1)
    in_memory_db.commit()

    row2 = PriceHistory(ticker="TEST", date="2022-01-01", close=200.0)
    in_memory_db.add(row2)
    with pytest.raises(IntegrityError):
        in_memory_db.commit()


# ---------------------------------------------------------------------------
# Edge case
# ---------------------------------------------------------------------------

def test_backtest_run_repr() -> None:
    """BacktestRun __repr__ contains strategy_name."""
    run = BacktestRun(
        strategy_name="Markowitz",
        run_at="2024-01-01T00:00:00",
        tickers='["AAPL","MSFT"]',
        sharpe=1.5,
        max_drawdown=-0.1,
        total_return=0.25,
    )
    assert "Markowitz" in repr(run)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_corporation_requires_ticker(in_memory_db: Session) -> None:
    """Corporation without a ticker should fail on commit."""
    from sqlalchemy.exc import IntegrityError

    corp = Corporation(name="No Ticker Corp")
    in_memory_db.add(corp)
    with pytest.raises(Exception):  # IntegrityError or StatementError
        in_memory_db.commit()
