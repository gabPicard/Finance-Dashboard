"""Data fetching with a three-layer lookup: cache → SQLite → external API stub.

The public surface is :func:`get_prices`.  Legacy helpers (``fetch_stock_data``,
``fetch_stock_info``, etc.) are retained for backward compatibility with existing
pipeline and strategy code.
"""

from __future__ import annotations

import os

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from .cache import default_cache

load_dotenv()


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DataUnavailableError(RuntimeError):
    """Raised when price data cannot be retrieved from any layer."""


# ---------------------------------------------------------------------------
# Three-layer public entry-point
# ---------------------------------------------------------------------------


def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return close prices for *ticker* over [*start*, *end*].

    Lookup order
    ------------
    1. **In-memory cache** – checked first; a hit avoids any I/O.
    2. **SQLite database** – checked via :func:`src.db.database.get_prices_from_db`.
    3. **External API** – stub; see inline comment for where to add the real call.

    Parameters
    ----------
    ticker:
        Asset ticker symbol (e.g. ``"AAPL"``).
    start:
        ISO-8601 start date string (e.g. ``"2022-01-01"``).
    end:
        ISO-8601 end date string (e.g. ``"2023-01-01"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex named ``"date"`` and a single column
        equal to *ticker* containing close prices.

    Raises
    ------
    DataUnavailableError
        When data is not found in any layer.
    """
    # ── Layer 1: cache ────────────────────────────────────────────────────
    cached = default_cache.get(ticker, start, end)
    if cached is not None:
        return cached

    # ── Layer 2: SQLite ───────────────────────────────────────────────────
    try:
        from src.db.database import get_prices_from_db  # avoid circular at module level

        db_data = get_prices_from_db(ticker, start, end)
        if db_data is not None and not db_data.empty:
            default_cache.set(ticker, start, end, db_data)
            return db_data
    except Exception:
        pass  # DB not available – fall through to API layer

    # ── Layer 3: External API (STUB) ──────────────────────────────────────
    # TODO: Replace the body of this block with a real Alpaca API call.
    #
    # Example (Alpaca):
    #   from alpaca.data.historical import StockHistoricalDataClient
    #   from alpaca.data.requests import StockBarsRequest
    #   from alpaca.data.timeframe import TimeFrame
    #
    #   client = StockHistoricalDataClient(
    #       api_key=os.getenv("ALPACA_API_KEY"),
    #       secret_key=os.getenv("ALPACA_SECRET_KEY"),
    #   )
    #   request = StockBarsRequest(
    #       symbol_or_symbols=ticker,
    #       timeframe=TimeFrame.Day,
    #       start=start,
    #       end=end,
    #   )
    #   bars = client.get_stock_bars(request).df
    #   df = bars[["close"]].rename(columns={"close": ticker})
    #   df.index.name = "date"
    #   default_cache.set(ticker, start, end, df)
    #   return df
    raise DataUnavailableError(
        f"No price data found for {ticker!r} between {start!r} and {end!r}. "
        "Implement an API call in the Layer 3 stub in src/data/fetch_data.py."
    )


# ---------------------------------------------------------------------------
# Legacy helpers (yfinance-backed)  – kept for backward compatibility
# ---------------------------------------------------------------------------


def fetch_stock_data(
    tickers: str | list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV data via yfinance.

    Parameters
    ----------
    tickers:
        Single ticker string or list of tickers.
    start_date:
        Optional start date (ISO-8601).  When ``None`` the *period* argument is
        used instead.
    end_date:
        Optional end date (ISO-8601).  Defaults to today when ``None``.
    period:
        yfinance period string (e.g. ``"1y"``).  Used only when *start_date*
        is ``None``.
    interval:
        yfinance interval string (e.g. ``"1d"``).

    Returns
    -------
    pd.DataFrame
        Raw yfinance DataFrame (potentially MultiIndex columns for multiple
        tickers).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    if start_date and end_date:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
    else:
        data = yf.download(
            tickers,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
    return data


def fetch_stock_info(tickers: str | list[str]) -> dict:
    """Fetch metadata for one or more tickers via yfinance.

    Parameters
    ----------
    tickers:
        Single ticker string or list of tickers.

    Returns
    -------
    dict
        Mapping of ticker → info dict (or ``None`` on failure).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    stock_info: dict = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_info[ticker] = stock.info
        except Exception as exc:
            print(f"Error fetching info for {ticker}: {exc}")
            stock_info[ticker] = None
    return stock_info


def fetch_risk_free_rate(ticker: str = "^FVX") -> float:
    """Fetch the current risk-free rate from a Treasury yield ticker.

    Returns 4 % (0.04) as fallback if the call fails or the value looks
    unreasonable.

    Parameters
    ----------
    ticker:
        Treasury yield ticker symbol.

    Returns
    -------
    float
        Annualised risk-free rate as a decimal (e.g. ``0.045`` for 4.5 %).
    """
    try:
        treasury = yf.Ticker(ticker)
        hist = treasury.history(period="5d")
        if not hist.empty:
            rate = hist["Close"].iloc[-1]
            if rate > 1:
                rate = rate / 100
            if rate < 0 or rate > 0.20:
                return 0.04
            return float(rate)
    except Exception:
        pass
    return 0.04


def get_company_info(ticker: str, info: list[str]) -> dict | None:
    """Retrieve selected metadata fields for a single ticker.

    Parameters
    ----------
    ticker:
        Asset ticker symbol.
    info:
        List of yfinance info keys to extract.

    Returns
    -------
    dict | None
        Mapping of requested keys to values, or ``None`` on failure.
    """
    result: dict = {}
    try:
        raw = fetch_stock_info(ticker).get(ticker, {}) or {}
        for key in info:
            result[key] = raw.get(key)
    except Exception as exc:
        print(f"Error fetching info for {ticker}: {exc}")
        return None
    return result


def get_company_name(ticker: str) -> str:
    """Return the display name for *ticker*, falling back to the ticker itself.

    Parameters
    ----------
    ticker:
        Asset ticker symbol.

    Returns
    -------
    str
        Human-readable company name.
    """
    try:
        info = fetch_stock_info(ticker).get(ticker, {}) or {}
        return (
            info.get("displayName")
            or info.get("shortName")
            or info.get("longName")
            or ticker
        )
    except Exception:
        return ticker


def get_company_sector(ticker: str) -> str:
    """Return the sector for *ticker*, falling back to ``"Unknown"``.

    Parameters
    ----------
    ticker:
        Asset ticker symbol.

    Returns
    -------
    str
        Sector string.
    """
    try:
        info = fetch_stock_info(ticker).get(ticker, {}) or {}
        return (
            info.get("sector")
            or info.get("sectorDisp")
            or info.get("sectorKey")
            or "Unknown"
        )
    except Exception:
        return "Unknown"
