"""Portfolio dataclass — the standard result object for allocation strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Portfolio:
    """Holds the result of a portfolio allocation strategy run.

    Attributes
    ----------
    name:
        Human-readable strategy name.
    tickers:
        Ordered list of ticker symbols.
    weights:
        Mapping of ticker → current weight (should sum to 1.0).
    weights_history:
        DataFrame recording weights at each rebalancing date (index=date,
        columns=tickers).  Populated by ``backtest()``.
    returns_history:
        Series of daily portfolio returns over the backtest period.
    metrics:
        Arbitrary performance metrics dict (sharpe, max_drawdown, etc.).
    """

    name: str
    tickers: list[str]
    weights: dict[str, float] = field(default_factory=dict)
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns_history: pd.Series = field(default_factory=pd.Series)
    metrics: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Assert invariants on the portfolio.

        Raises
        ------
        ValueError
            When weights do not sum to 1.0 (within float tolerance) or when
            any ticker is missing from the weights dict.
        """
        if not self.weights:
            raise ValueError("Portfolio has no weights.")

        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"Portfolio weights sum to {total:.8f}, expected 1.0 (±1e-6)."
            )

        missing = [t for t in self.tickers if t not in self.weights]
        if missing:
            raise ValueError(
                f"The following tickers are missing from weights: {missing}"
            )

    def weight_series(self) -> pd.Series:
        """Return the current weights as a pandas Series indexed by ticker.

        Returns
        -------
        pd.Series
            Weight values indexed by ticker.
        """
        return pd.Series(self.weights, name="weight")

    def __repr__(self) -> str:
        return (
            f"Portfolio(name={self.name!r}, tickers={self.tickers}, "
            f"weights={self.weights})"
        )
