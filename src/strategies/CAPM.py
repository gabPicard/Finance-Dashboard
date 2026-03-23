"""CAPM-based portfolio allocation strategy.

Estimates beta for each asset against a market index and computes CAPM
expected returns.  The portfolio is weighted proportionally to expected return
(long-only, normalised to sum to 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.data_engineering import simple_returns
from ..results.portfolio import Portfolio
from .base_allocation import AllocationStrategy


# ---------------------------------------------------------------------------
# Low-level helpers (also usable from pipeline code)
# ---------------------------------------------------------------------------


def calculate_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """Compute the beta of an asset relative to a market index.

    Parameters
    ----------
    asset_returns:
        Historical returns of the asset.
    market_returns:
        Historical returns of the market benchmark.

    Returns
    -------
    float
        Beta coefficient.
    """
    aligned = pd.DataFrame({"asset": asset_returns, "market": market_returns}).dropna()
    if len(aligned) < 2:
        return 1.0
    mkt_var = aligned["market"].var(ddof=1)
    if mkt_var == 0:
        return 1.0
    return float(aligned["asset"].cov(aligned["market"]) / mkt_var)


def capm_expected_returns(
    prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> dict[str, float]:
    """Compute CAPM expected returns for each asset.

    Formula: E[R_i] = R_f + β_i × (E[R_m] − R_f)

    Parameters
    ----------
    prices:
        Asset price DataFrame (index=date, columns=tickers).
    market_prices:
        Market index price DataFrame.
    risk_free_rate:
        Annualised risk-free rate (default 4 %).

    Returns
    -------
    dict[str, float]
        Mapping of ticker → CAPM expected annual return.
    """
    asset_rets = simple_returns(prices).dropna()
    mkt_rets = simple_returns(market_prices).dropna()

    if isinstance(mkt_rets, pd.DataFrame):
        mkt_series = mkt_rets.iloc[:, 0]
    else:
        mkt_series = mkt_rets

    mkt_expected = float(mkt_series.mean() * 252)
    result: dict[str, float] = {}

    for ticker in asset_rets.columns:
        beta_i = calculate_beta(asset_rets[ticker], mkt_series)
        result[ticker] = float(risk_free_rate + beta_i * (mkt_expected - risk_free_rate))

    return result


# ---------------------------------------------------------------------------
# CAPM Strategy class
# ---------------------------------------------------------------------------


class CAPMStrategy(AllocationStrategy):
    """CAPM-weighted portfolio allocation.

    Weights are proportional to CAPM expected returns (positive returns only,
    normalised to sum to 1).

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
    market_prices:
        Market index price DataFrame.
    risk_free_rate:
        Annualised risk-free rate (default 0.04).
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
        market_prices: pd.DataFrame,
        risk_free_rate: float = 0.04,
        window: int = 252,
    ) -> None:
        """Initialise with price data."""
        super().__init__(name, tickers, start, end)
        self.prices = prices
        self.market_prices = market_prices
        self.risk_free_rate = risk_free_rate
        self.window = window

    def run(self) -> Portfolio:
        """Compute CAPM weights on the full data window.

        Returns
        -------
        Portfolio
            Portfolio with ``weights`` set to normalised CAPM expected returns.
        """
        exp_returns = capm_expected_returns(self.prices, self.market_prices, self.risk_free_rate)

        # Keep only assets with positive expected return
        positive = {k: v for k, v in exp_returns.items() if v > 0}
        if not positive:
            n = len(self.tickers)
            positive = {t: 1.0 / n for t in self.tickers}

        total = sum(positive.values())
        weights_dict = {k: v / total for k, v in positive.items()}

        portfolio = Portfolio(
            name=self.name,
            tickers=list(weights_dict.keys()),
            weights=weights_dict,
        )
        self._last_result = portfolio
        return portfolio

    def backtest(self) -> Portfolio:
        """Rolling CAPM backtest.

        Returns
        -------
        Portfolio
            Portfolio with ``weights_history`` and ``returns_history`` populated.
        """
        if "date" in self.prices.columns:
            prices_idx = self.prices.set_index("date")
        else:
            prices_idx = self.prices.copy()
        prices_idx.index = pd.to_datetime(prices_idx.index)
        prices_idx = prices_idx.sort_index()

        if "date" in self.market_prices.columns:
            mkt_idx = self.market_prices.set_index("date")
        else:
            mkt_idx = self.market_prices.copy()
        mkt_idx.index = pd.to_datetime(mkt_idx.index)
        mkt_idx = mkt_idx.sort_index()

        rebalance_dates = prices_idx.resample("QE").last().index
        asset_cols = list(prices_idx.columns)
        weights_records: list[dict] = []

        for date in rebalance_dates:
            window_p = prices_idx.loc[:date].tail(self.window)
            window_m = mkt_idx.loc[:date].tail(self.window)

            if len(window_p) < self.window // 2:
                continue

            exp_ret = capm_expected_returns(window_p, window_m, self.risk_free_rate)
            positive = {k: v for k, v in exp_ret.items() if v > 0}
            if not positive:
                continue

            total = sum(positive.values())
            row: dict = {col: 0.0 for col in asset_cols}
            for ticker, er in positive.items():
                if ticker in asset_cols:
                    row[ticker] = er / total
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
