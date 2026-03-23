"""StockPrices: convenience wrapper around the fetch / data-engineering stack."""

from __future__ import annotations

import json
import os

import pandas as pd

from .data_engineering import (
    annualised_volatility,
    delete_assets,
    fix_price_anomalies,
    log_returns,
    rolling_correlation,
    rolling_covariance,
    simple_returns,
    validate_and_clean_data,
)
from .fetch_data import fetch_risk_free_rate, fetch_stock_data


def get_tickers_list(market: str) -> tuple[list[str], str, str]:
    """Load tickers, risk-free-rate ticker, and market index ticker from JSON.

    Parameters
    ----------
    market:
        Market key as it appears in ``tickers_list.json`` (e.g. ``"S&P500"``).

    Returns
    -------
    tuple[list[str], str, str]
        ``(tickers, rfr_ticker, market_ticker)``
    """
    json_path = os.path.join(os.path.dirname(__file__), "tickers_list.json")
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    market_data = data[market]
    tickers: list[str] = market_data["Tickers list"]
    rfr = market_data["Risk free rate"]
    if isinstance(rfr, list):
        rfr = rfr[0]
    market_ticker: str = market_data["Market ticker"]
    return tickers, rfr, market_ticker


def merge_markets(market_list: list[str]) -> list[str]:
    """Merge tickers from multiple markets into a de-duplicated list.

    Parameters
    ----------
    market_list:
        List of market keys.

    Returns
    -------
    list[str]
        Merged, de-duplicated list of tickers.
    """
    merged: list[str] = []
    for market in market_list:
        try:
            tickers, _, _ = get_tickers_list(market)
            for t in tickers:
                if t not in merged:
                    merged.append(t)
        except Exception as exc:
            print(f"Error fetching tickers for {market}: {exc}")
    return merged


def get_stock_prices(
    market: str | list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "1y",
    interval: str = "1d",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Fetch and clean stock prices for one or more markets.

    Parameters
    ----------
    market:
        Single market name or list of market names.
    start_date:
        Optional start date (ISO-8601).
    end_date:
        Optional end date (ISO-8601).
    period:
        yfinance period string (used when *start_date* is ``None``).
    interval:
        yfinance interval string.
    columns:
        OHLCV columns to extract (default ``["Close"]``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, float]
        ``(clean_prices, clean_market_prices, risk_free_rate)``
    """
    if columns is None:
        columns = ["Close"]

    if isinstance(market, list):
        tickers_list = merge_markets(market)
        rfr_ticker = "^IRX"
        market_ticker = "^GSPC"
    else:
        tickers_list, rfr_ticker, market_ticker = get_tickers_list(market)

    raw_prices = fetch_stock_data(tickers_list, start_date=start_date, end_date=end_date,
                                  period=period, interval=interval)
    raw_market = fetch_stock_data([market_ticker], start_date=start_date, end_date=end_date,
                                  period=period, interval=interval)
    risk_free_rate = fetch_risk_free_rate(rfr_ticker)

    def _process(raw: pd.DataFrame) -> pd.DataFrame:
        tmp = raw[columns]
        if isinstance(tmp.columns, pd.MultiIndex):
            tmp.columns = [col[1] if col[1] else col[0] for col in tmp.columns]
        tmp, _ = fix_price_anomalies(tmp)
        clean, excluded = validate_and_clean_data(tmp)
        reset = clean.reset_index()
        first_col = reset.columns[0]
        if first_col in ("index", "Date", "Datetime"):
            reset = reset.rename(columns={first_col: "date"})
        elif "date" not in reset.columns:
            reset.insert(0, "date", clean.index)
        return reset

    clean_prices = _process(raw_prices)
    clean_market = _process(raw_market)

    # Remove any excluded assets from the ticker JSON
    if isinstance(market, str):
        all_excluded = [
            col for col in tickers_list
            if col not in clean_prices.columns
        ]
        if all_excluded:
            delete_assets(all_excluded, market)

    return clean_prices, clean_market, risk_free_rate


class StockPrices:
    """Convenience wrapper that loads prices and exposes analysis methods.

    Parameters
    ----------
    market:
        Single market name or list of market names.
    start_date:
        Optional start date (ISO-8601).
    end_date:
        Optional end date (ISO-8601).
    period:
        yfinance period string (used when *start_date* is ``None``).
    """

    def __init__(
        self,
        market: str | list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        period: str = "1y",
    ) -> None:
        """Initialise and fetch prices."""
        prices_df, market_df, rfr = get_stock_prices(
            market, start_date=start_date, end_date=end_date, period=period
        )
        # Convert to clean price DataFrame (DatetimeIndex, tickers as columns)
        self._prices = self._to_indexed(prices_df)
        self._market_prices = self._to_indexed(market_df)
        self.risk_free_rate: float = rfr

    # ------------------------------------------------------------------

    @staticmethod
    def _to_indexed(df: pd.DataFrame) -> pd.DataFrame:
        """Set the 'date' column as DatetimeIndex if present."""
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    @property
    def prices(self) -> pd.DataFrame:
        """Raw price DataFrame (index=date, columns=tickers)."""
        return self._prices

    @property
    def market_prices(self) -> pd.DataFrame:
        """Market index price DataFrame."""
        return self._market_prices

    def get_returns(self, log: bool = False) -> pd.DataFrame:
        """Return the returns DataFrame.

        Parameters
        ----------
        log:
            Use log returns when ``True`` (default: simple returns).
        """
        if log:
            return log_returns(self._prices)
        return simple_returns(self._prices)

    def get_covariance(self, window: int = 252) -> pd.DataFrame:
        """Return the rolling covariance matrix.

        Parameters
        ----------
        window:
            Rolling window in trading days.
        """
        return rolling_covariance(self._prices, window=window)

    def get_correlation(self, window: int = 252) -> pd.DataFrame:
        """Return the rolling correlation matrix.

        Parameters
        ----------
        window:
            Rolling window in trading days.
        """
        return rolling_correlation(self._prices, window=window)

    def get_volatility(self, window: int | None = None) -> pd.Series:
        """Return annualised volatility per ticker.

        Parameters
        ----------
        window:
            Optional rolling window.  ``None`` uses the full history.
        """
        return annualised_volatility(self._prices, window=window)
