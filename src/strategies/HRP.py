"""Hierarchical Risk Parity (HRP) portfolio allocation strategy.

Steps:
1. Compute the correlation matrix from returns.
2. Build a distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij)).
3. Hierarchical clustering (Ward linkage via scipy).
4. Recursive bisection to assign weights inversely proportional to variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ..data.data_engineering import simple_returns
from ..results.portfolio import Portfolio
from .base_allocation import AllocationStrategy


# ---------------------------------------------------------------------------
# Core HRP helpers
# ---------------------------------------------------------------------------


def _correlation_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix to a distance matrix.

    Distance: d_ij = sqrt(0.5 × (1 − ρ_ij)).

    Parameters
    ----------
    corr:
        Correlation matrix DataFrame.

    Returns
    -------
    pd.DataFrame
        Distance matrix of the same shape.
    """
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0.0, 1.0))
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def _quasi_diagonalise(link: np.ndarray, n: int) -> list[int]:
    """Return asset indices in quasi-diagonal (dendrogram) order.

    Parameters
    ----------
    link:
        Scipy linkage matrix (n-1 × 4).
    n:
        Number of original assets.

    Returns
    -------
    list[int]
        Reordered asset indices.
    """
    link_int = link.astype(int)
    sorted_items: list[int] = list(range(n))

    def _get_cluster_items(cluster_id: int) -> list[int]:
        if cluster_id < n:
            return [cluster_id]
        row = link_int[cluster_id - n]
        left = _get_cluster_items(row[0])
        right = _get_cluster_items(row[1])
        return left + right

    return _get_cluster_items(n + len(link_int) - 1)


def _recursive_bisection(
    cov: pd.DataFrame, sorted_items: list[int]
) -> dict[int, float]:
    """Assign weights via recursive bisection.

    Parameters
    ----------
    cov:
        Covariance matrix indexed by position (0-based).
    sorted_items:
        Asset indices in quasi-diagonal order.

    Returns
    -------
    dict[int, float]
        Mapping of asset index → weight.
    """
    weights: dict[int, float] = {i: 1.0 for i in sorted_items}

    def _cluster_variance(items: list[int]) -> float:
        sub_cov = cov.iloc[items, items]
        ivp = 1.0 / np.diag(sub_cov.values)
        ivp /= ivp.sum()
        return float(ivp @ sub_cov.values @ ivp)

    def _recurse(items: list[int]) -> None:
        if len(items) == 1:
            return
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]
        var_left = _cluster_variance(left)
        var_right = _cluster_variance(right)
        alpha = 1.0 - var_left / (var_left + var_right)  # right gets alpha
        for i in left:
            weights[i] *= 1.0 - alpha
        for i in right:
            weights[i] *= alpha
        _recurse(left)
        _recurse(right)

    _recurse(sorted_items)
    return weights


def hrp_weights(
    returns: pd.DataFrame,
) -> pd.Series:
    """Compute HRP weights for a returns DataFrame.

    Parameters
    ----------
    returns:
        Returns DataFrame (index=date, columns=tickers).

    Returns
    -------
    pd.Series
        Weight Series indexed by ticker.
    """
    corr = returns.corr()
    cov = returns.cov() * 252

    dist = _correlation_to_distance(corr)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="ward")

    n = len(returns.columns)
    sorted_items = _quasi_diagonalise(link, n)

    cov_indexed = pd.DataFrame(
        cov.values,
        index=range(n),
        columns=range(n),
    )
    weight_map = _recursive_bisection(cov_indexed, sorted_items)

    tickers = list(returns.columns)
    weights = pd.Series(
        {tickers[i]: weight_map.get(i, 0.0) for i in range(n)},
        name="weight",
    )
    # Normalise
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


# ---------------------------------------------------------------------------
# HRP Strategy class
# ---------------------------------------------------------------------------


class HRPStrategy(AllocationStrategy):
    """Hierarchical Risk Parity allocation strategy.

    Parameters
    ----------
    name:
        Strategy name.
    tickers:
        Asset tickers.
    start:
        ISO-8601 start date.
    end:
        ISO-8601 end date.
    prices:
        Asset price DataFrame (index=date, columns=tickers).
    window:
        Rolling backtest window in days (default 252).
    """

    def __init__(
        self,
        name: str,
        tickers: list[str],
        start: str,
        end: str,
        prices: pd.DataFrame,
        window: int = 252,
    ) -> None:
        """Initialise with price data."""
        super().__init__(name, tickers, start, end)
        self.prices = prices
        self.window = window

    def run(self) -> Portfolio:
        """Compute HRP weights on the full data window.

        Returns
        -------
        Portfolio
            Portfolio with ``weights`` set to HRP allocation.
        """
        rets = simple_returns(self.prices).dropna()
        weights_series = hrp_weights(rets)

        portfolio = Portfolio(
            name=self.name,
            tickers=list(weights_series.index),
            weights=weights_series.to_dict(),
        )
        self._last_result = portfolio
        return portfolio

    def backtest(self) -> Portfolio:
        """Rolling-window HRP backtest.

        Returns
        -------
        Portfolio
            Portfolio with ``weights_history`` populated.
        """
        if "date" in self.prices.columns:
            prices_idx = self.prices.set_index("date")
        else:
            prices_idx = self.prices.copy()
        prices_idx.index = pd.to_datetime(prices_idx.index)
        prices_idx = prices_idx.sort_index()

        rebalance_dates = prices_idx.resample("QE").last().index
        asset_cols = list(prices_idx.columns)
        weights_records: list[dict] = []

        for date in rebalance_dates:
            window_p = prices_idx.loc[:date].tail(self.window)
            if len(window_p) < self.window // 2:
                continue

            rets = simple_returns(window_p).dropna()
            if rets.shape[1] < 2:
                continue

            try:
                w = hrp_weights(rets)
            except Exception:
                continue

            row: dict = {col: 0.0 for col in asset_cols}
            for ticker, weight in w.items():
                if ticker in asset_cols:
                    row[ticker] = float(weight)
            row["date"] = date
            weights_records.append(row)

        if not weights_records:
            return Portfolio(name=self.name, tickers=self.tickers)

        weights_df = pd.DataFrame(weights_records).set_index("date")
        portfolio = Portfolio(
            name=self.name,
            tickers=asset_cols,
            weights={col: float(weights_df[col].iloc[-1]) for col in asset_cols},
            weights_history=weights_df,
        )
        self._last_result = portfolio
        return portfolio
