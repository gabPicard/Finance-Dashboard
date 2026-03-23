"""TradingBook dataclass — the standard result object for trading strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TradingBook:
    """Holds the result of a pair-trading (or other market-neutral) strategy run.

    Attributes
    ----------
    name:
        Human-readable strategy name.
    tickers:
        Ordered list of ticker symbols involved.
    positions:
        Mapping of ticker → signed position size.  Positive = long,
        negative = short.
    positions_history:
        DataFrame recording positions at each time step (index=date,
        columns=tickers).  Populated by ``backtest()``.
    pnl_history:
        Series of daily mark-to-market P&L.
    metrics:
        Arbitrary performance metrics dict (sharpe, max_drawdown, etc.).
    """

    name: str
    tickers: list[str]
    positions: dict[str, float] = field(default_factory=dict)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    pnl_history: pd.Series = field(default_factory=pd.Series)
    metrics: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def net_exposure(self) -> float:
        """Return the sum of absolute position values.

        Returns
        -------
        float
            Gross exposure (sum of |position| for all tickers).
        """
        return sum(abs(v) for v in self.positions.values())

    def net_position(self) -> float:
        """Return the algebraic sum of all positions (net directional exposure).

        Returns
        -------
        float
            Net signed position.
        """
        return sum(self.positions.values())

    def position_series(self) -> pd.Series:
        """Return the current positions as a pandas Series indexed by ticker.

        Returns
        -------
        pd.Series
            Signed position values indexed by ticker.
        """
        return pd.Series(self.positions, name="position")

    def __repr__(self) -> str:
        return (
            f"TradingBook(name={self.name!r}, tickers={self.tickers}, "
            f"net_exposure={self.net_exposure():.4f})"
        )
