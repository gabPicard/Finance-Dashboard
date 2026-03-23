"""Tests for src.db.database — helper functions."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Corporation, Market


@pytest.fixture
def db_engine():
    """In-memory SQLite engine with schema created."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_get_prices_from_db_returns_none_when_empty(db_engine) -> None:
    """get_prices_from_db returns None when no rows exist for a ticker."""
    import src.db.database as db
    original_engine = db._ENGINE
    original_factory = db._SessionFactory

    db._ENGINE = db_engine
    db._SessionFactory = sessionmaker(bind=db_engine)
    try:
        result = db.get_prices_from_db("AAPL", "2022-01-01", "2022-12-31")
        assert result is None
    finally:
        db._ENGINE = original_engine
        db._SessionFactory = original_factory


def test_insert_and_retrieve_prices(db_engine) -> None:
    """insert_prices + get_prices_from_db round-trip."""
    import src.db.database as db
    original_engine = db._ENGINE
    original_factory = db._SessionFactory
    db._ENGINE = db_engine
    db._SessionFactory = sessionmaker(bind=db_engine)

    # Seed market and corporation
    Session = sessionmaker(bind=db_engine)
    session = Session()
    mkt = Market(name="Test", exchange="Test", region="US", currency="USD")
    session.add(mkt)
    session.flush()
    corp = Corporation(ticker="AAPL", name="Apple", market_id=mkt.id)
    session.add(corp)
    session.commit()
    session.close()

    idx = pd.date_range("2022-01-03", periods=5, freq="B")
    df = pd.DataFrame({"close": [150.0, 151.0, 149.0, 152.0, 153.0]}, index=idx)

    try:
        db.insert_prices("AAPL", df)
        result = db.get_prices_from_db("AAPL", "2022-01-01", "2022-12-31")
        assert result is not None
        assert len(result) == 5
        assert "AAPL" in result.columns
    finally:
        db._ENGINE = original_engine
        db._SessionFactory = original_factory


# ---------------------------------------------------------------------------
# Edge case
# ---------------------------------------------------------------------------

def test_get_corporation_returns_none_for_unknown(db_engine) -> None:
    """get_corporation returns None for an unknown ticker."""
    import src.db.database as db
    original_engine = db._ENGINE
    original_factory = db._SessionFactory
    db._ENGINE = db_engine
    db._SessionFactory = sessionmaker(bind=db_engine)
    try:
        result = db.get_corporation("UNKNOWN_TICKER_XYZ")
        assert result is None
    finally:
        db._ENGINE = original_engine
        db._SessionFactory = original_factory


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_insert_prices_empty_df(db_engine) -> None:
    """insert_prices with empty DataFrame does nothing (no crash)."""
    import src.db.database as db
    original_engine = db._ENGINE
    original_factory = db._SessionFactory
    db._ENGINE = db_engine
    db._SessionFactory = sessionmaker(bind=db_engine)
    try:
        db.insert_prices("AAPL", pd.DataFrame())
    finally:
        db._ENGINE = original_engine
        db._SessionFactory = original_factory
