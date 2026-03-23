"""Tests for src.db.seed — database seeding idempotency."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Corporation, Market


@pytest.fixture
def seeded_session():
    """In-memory session with seed data applied."""
    import src.db.database as db
    from src.db.seed import seed

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    original_engine = db._ENGINE
    original_factory = db._SessionFactory
    db._ENGINE = engine
    db._SessionFactory = Session

    seed()

    session = Session()
    yield session
    session.close()

    db._ENGINE = original_engine
    db._SessionFactory = original_factory
    Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_markets_seeded(seeded_session) -> None:
    """At least 3 markets are created by seed()."""
    count = seeded_session.query(Market).count()
    assert count >= 3


def test_corporations_seeded(seeded_session) -> None:
    """At least 10 corporations are created by seed()."""
    count = seeded_session.query(Corporation).count()
    assert count >= 10


def test_known_ticker_exists(seeded_session) -> None:
    """AAPL should be present after seeding."""
    corp = seeded_session.query(Corporation).filter_by(ticker="AAPL").first()
    assert corp is not None
    assert corp.sector == "Technology"


# ---------------------------------------------------------------------------
# Edge case — idempotency
# ---------------------------------------------------------------------------

def test_seed_idempotent(seeded_session) -> None:
    """Calling seed() twice does not create duplicate rows."""
    import src.db.database as db
    from src.db.seed import seed

    count_before = seeded_session.query(Corporation).count()
    seeded_session.close()

    seed()
    seeded_session2 = sessionmaker(bind=seeded_session.bind)()
    count_after = seeded_session2.query(Corporation).count()
    seeded_session2.close()

    assert count_after == count_before


# ---------------------------------------------------------------------------
# Invalid inputs / structure
# ---------------------------------------------------------------------------

def test_corporation_has_market_fk(seeded_session) -> None:
    """All seeded corporations with a market_name have a valid market_id FK."""
    corps = seeded_session.query(Corporation).all()
    market_ids = {m.id for m in seeded_session.query(Market).all()}
    for corp in corps:
        if corp.market_id is not None:
            assert corp.market_id in market_ids
