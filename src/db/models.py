"""SQLAlchemy ORM models for the Finance Dashboard database."""

from __future__ import annotations

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


class Market(Base):
    """Represents a financial market / exchange."""

    __tablename__ = "markets"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String(100), nullable=False, unique=True)
    exchange: str = Column(String(100), nullable=False)
    region: str = Column(String(100), nullable=False)
    currency: str = Column(String(10), nullable=False)

    corporations: list[Corporation] = relationship(
        "Corporation", back_populates="market", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Market(id={self.id}, name={self.name!r}, exchange={self.exchange!r})>"


class Corporation(Base):
    """Represents a publicly listed corporation."""

    __tablename__ = "corporations"

    ticker: str = Column(String(20), primary_key=True)
    name: str = Column(String(200), nullable=False)
    sector: str | None = Column(String(100))
    industry: str | None = Column(String(200))
    market_cap: float | None = Column(Float)
    market_id: int | None = Column(Integer, ForeignKey("markets.id", ondelete="SET NULL"))
    description: str | None = Column(Text)
    ipo_date: str | None = Column(String(20))

    market: Market | None = relationship("Market", back_populates="corporations")
    price_history: list[PriceHistory] = relationship(
        "PriceHistory", back_populates="corporation", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_corporations_market_id", "market_id"),)

    def __repr__(self) -> str:
        return f"<Corporation(ticker={self.ticker!r}, name={self.name!r})>"


class PriceHistory(Base):
    """Daily OHLCV price record for a corporation."""

    __tablename__ = "price_history"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    ticker: str = Column(String(20), ForeignKey("corporations.ticker", ondelete="CASCADE"), nullable=False)
    date: str = Column(String(20), nullable=False)
    open: float | None = Column(Float)
    high: float | None = Column(Float)
    low: float | None = Column(Float)
    close: float = Column(Float, nullable=False)
    volume: float | None = Column(Float)

    corporation: Corporation = relationship("Corporation", back_populates="price_history")

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_price_history_ticker_date"),
        Index("ix_price_history_ticker", "ticker"),
        Index("ix_price_history_date", "date"),
    )

    def __repr__(self) -> str:
        return f"<PriceHistory(ticker={self.ticker!r}, date={self.date!r}, close={self.close})>"


class BacktestRun(Base):
    """Metadata record for a completed backtest run."""

    __tablename__ = "backtest_runs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name: str = Column(String(100), nullable=False)
    run_at: str = Column(String(30), nullable=False)
    tickers: str = Column(Text, nullable=False)  # JSON-encoded list
    sharpe: float | None = Column(Float)
    max_drawdown: float | None = Column(Float)
    total_return: float | None = Column(Float)

    __table_args__ = (Index("ix_backtest_runs_strategy_name", "strategy_name"),)

    def __repr__(self) -> str:
        return (
            f"<BacktestRun(id={self.id}, strategy={self.strategy_name!r}, "
            f"run_at={self.run_at!r})>"
        )
