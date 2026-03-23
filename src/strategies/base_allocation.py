"""Abstract base class for all portfolio allocation strategies."""

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
from ..results.portfolio import Portfolio


class AllocationStrategy(ABC):
    """Abstract base class for portfolio allocation strategies.

    Subclasses must implement :meth:`run` and :meth:`backtest`.

    Parameters
    ----------
    name:
        Human-readable strategy name.
    tickers:
        List of asset tickers to include in the portfolio.
    start:
        ISO-8601 start date for the data window (e.g. ``"2020-01-01"``).
    end:
        ISO-8601 end date for the data window (e.g. ``"2023-12-31"``).
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
        self._last_result: Portfolio | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> Portfolio:
        """Compute optimal weights and return a :class:`Portfolio`.

        Returns
        -------
        Portfolio
            Result with ``weights`` populated.
        """

    @abstractmethod
    def backtest(self) -> Portfolio:
        """Run a rolling-window backtest and return a :class:`Portfolio`.

        Returns
        -------
        Portfolio
            Result with ``weights_history`` and ``returns_history`` populated.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Compute performance metrics on the last :meth:`run` / :meth:`backtest` result.

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
        rets = self._last_result.returns_history
        if rets.empty:
            return {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "annualised_return": 0.0,
                "annualised_volatility": 0.0,
                "var_95": 0.0,
            }
        return {
            "sharpe": sharpe_ratio(rets),
            "max_drawdown": max_drawdown(rets),
            "annualised_return": annualised_return(rets),
            "annualised_volatility": annualised_volatility(rets),
            "var_95": value_at_risk(rets, confidence=0.95),
        }
