"""SQLAlchemy engine, session setup, and database helpers."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Generator

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Corporation, PriceHistory

load_dotenv()

_DB_PATH: str = os.getenv("DB_PATH", "data/finance.db")
_ENGINE = create_engine(
    f"sqlite:///{_DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)
_SessionFactory = sessionmaker(bind=_ENGINE, autoflush=True, autocommit=False)


def get_session() -> Session:
    """Return a new SQLAlchemy Session bound to the configured SQLite engine.

    The caller is responsible for committing and closing the session.
    """
    return _SessionFactory()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager that yields a session and handles commit/rollback."""
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables defined in the ORM models if they do not exist.

    Safe to call multiple times (idempotent).
    """
    os.makedirs(os.path.dirname(_DB_PATH) if os.path.dirname(_DB_PATH) else ".", exist_ok=True)
    Base.metadata.create_all(_ENGINE)


def get_prices_from_db(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily close prices for *ticker* between *start* and *end* from SQLite.

    Parameters
    ----------
    ticker:
        The asset ticker symbol.
    start:
        ISO-8601 start date string (e.g. ``"2023-01-01"``).
    end:
        ISO-8601 end date string (e.g. ``"2023-12-31"``).

    Returns
    -------
    pd.DataFrame | None
        DataFrame with index=date and columns=[ticker], or ``None`` if no rows
        are found.
    """
    with session_scope() as session:
        rows = (
            session.query(PriceHistory)
            .filter(
                PriceHistory.ticker == ticker,
                PriceHistory.date >= start,
                PriceHistory.date <= end,
            )
            .order_by(PriceHistory.date)
            .all()
        )

    if not rows:
        return None

    records = {r.date: r.close for r in rows}
    df = pd.DataFrame.from_dict(records, orient="index", columns=[ticker])
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def insert_prices(ticker: str, df: pd.DataFrame) -> None:
    """Insert or ignore OHLCV rows from *df* into the price_history table.

    Parameters
    ----------
    ticker:
        The asset ticker symbol.
    df:
        DataFrame with DatetimeIndex and at minimum a ``close`` column.
        Optional columns: ``open``, ``high``, ``low``, ``volume``.
    """
    with session_scope() as session:
        for date, row in df.iterrows():
            date_str = str(date)[:10]
            existing = (
                session.query(PriceHistory)
                .filter_by(ticker=ticker, date=date_str)
                .first()
            )
            if existing:
                continue
            record = PriceHistory(
                ticker=ticker,
                date=date_str,
                open=float(row.get("open", row.get("Open", None) or 0)) or None,
                high=float(row.get("high", row.get("High", None) or 0)) or None,
                low=float(row.get("low", row.get("Low", None) or 0)) or None,
                close=float(row.get("close", row.get("Close", row.iloc[0]))),
                volume=float(row.get("volume", row.get("Volume", None) or 0)) or None,
            )
            session.add(record)


def get_corporations_in_market(market_id: int) -> list[Corporation]:
    """Return all Corporation records that belong to the given *market_id*.

    Parameters
    ----------
    market_id:
        Primary key of the target market.

    Returns
    -------
    list[Corporation]
        Possibly empty list of Corporation ORM objects.
    """
    with session_scope() as session:
        corps = (
            session.query(Corporation)
            .filter(Corporation.market_id == market_id)
            .all()
        )
        # Detach from session so callers can use the objects freely
        session.expunge_all()
    return corps


def get_corporation(ticker: str) -> Corporation | None:
    """Return the Corporation with *ticker*, or ``None`` if not found.

    Parameters
    ----------
    ticker:
        The asset ticker symbol.

    Returns
    -------
    Corporation | None
    """
    with session_scope() as session:
        corp = session.query(Corporation).filter_by(ticker=ticker).first()
        if corp:
            session.expunge(corp)
    return corp
