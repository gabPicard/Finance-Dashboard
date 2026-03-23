"""Abstract base class for all trading (market-neutral) strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..metrics.portfolio_measurements import (
    annualised_return,
    annualised_volatility,
    max_drawdown,
    sharpe_ratio,
    value_at_risk,
)
from ..results.trading_book import TradingBook


class TradingStrategy(ABC):
    """Abstract base class for trading strategies.

    Subclasses must implement :meth:`run` and :meth:`backtest`.

    Parameters
    ----------
    name:
        Human-readable strategy name.
    tickers:
        List of asset tickers to trade.
    start:
        ISO-8601 start date (e.g. ``"2020-01-01"``).
    end:
        ISO-8601 end date (e.g. ``"2023-12-31"``).
    """

    def __init__(
        self,
        name: str,
        tickers: list[str],
        start: str,
        end: str,
    ) -> None:
        """Store strategy parameters."""
        self.name = name
        self.tickers = tickers
        self.start = start
        self.end = end
        self._last_result: TradingBook | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> TradingBook:
        """Generate current signals and return a :class:`TradingBook`.

        Returns
        -------
        TradingBook
            Result with ``positions`` populated.
        """

    @abstractmethod
    def backtest(self) -> TradingBook:
        """Run a full historical backtest and return a :class:`TradingBook`.

        Returns
        -------
        TradingBook
            Result with ``positions_history`` and ``pnl_history`` populated.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Compute performance metrics on the last run result.

        Returns
        -------
        dict[str, Any]
            Keys: ``sharpe``, ``max_drawdown``, ``annualised_return``,
            ``annualised_volatility``, ``var_95``.

        Raises
        ------
        RuntimeError
            When neither :meth:`run` nor :meth:`backtest` has been called yet.
        """
        if self._last_result is None:
            raise RuntimeError(
                "No result available. Call run() or backtest() first."
            )
        pnl = self._last_result.pnl_history
        if pnl.empty:
            return {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "annualised_return": 0.0,
                "annualised_volatility": 0.0,
                "var_95": 0.0,
            }
        return {
            "sharpe": sharpe_ratio(pnl),
            "max_drawdown": max_drawdown(pnl),
            "annualised_return": annualised_return(pnl),
            "annualised_volatility": annualised_volatility(pnl),
            "var_95": value_at_risk(pnl, confidence=0.95),
        }
