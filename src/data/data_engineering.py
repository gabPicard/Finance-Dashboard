"""Data-engineering helpers: returns, covariance, correlation, volatility.

All public functions accept a prices DataFrame with DatetimeIndex (rows = dates,
columns = tickers) and return a DataFrame or Series.  Helper utilities for data
cleaning and portfolio formatting are also included.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from .fetch_data import get_company_name, get_company_sector


# ---------------------------------------------------------------------------
# Core financial-engineering functions
# ---------------------------------------------------------------------------


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns: ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices:
        Price DataFrame (index=date, columns=tickers).

    Returns
    -------
    pd.DataFrame
        Log-return DataFrame of the same shape, with the first row dropped.
    """
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) returns: (P_t - P_{t-1}) / P_{t-1}.

    Parameters
    ----------
    prices:
        Price DataFrame (index=date, columns=tickers).

    Returns
    -------
    pd.DataFrame
        Simple-return DataFrame of the same shape, with the first row dropped.
    """
    return prices.pct_change(fill_method=None).dropna()


def rolling_covariance(
    prices: pd.DataFrame, window: int = 252, annualise: bool = True
) -> pd.DataFrame:
    """Compute the rolling covariance matrix over a sliding window.

    Parameters
    ----------
    prices:
        Price DataFrame (index=date, columns=tickers).
    window:
        Rolling window size in trading days (default 252).
    annualise:
        When ``True`` (default), multiply by 252 to annualise.

    Returns
    -------
    pd.DataFrame
        Rolling covariance evaluated at the last available date.
    """
    rets = simple_returns(prices)
    cov = rets.rolling(window).cov()
    if annualise:
        cov = cov * 252
    return cov


def rolling_correlation(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Compute the rolling correlation matrix over a sliding window.

    Parameters
    ----------
    prices:
        Price DataFrame (index=date, columns=tickers).
    window:
        Rolling window size in trading days (default 252).

    Returns
    -------
    pd.DataFrame
        Rolling correlation matrix evaluated at the last window.
    """
    rets = simple_returns(prices)
    return rets.rolling(window).corr()


def annualised_volatility(prices: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Compute annualised volatility for each ticker.

    Parameters
    ----------
    prices:
        Price DataFrame (index=date, columns=tickers).
    window:
        Optional rolling window.  When ``None`` (default), uses the full
        history.

    Returns
    -------
    pd.Series
        Annualised volatility indexed by ticker.
    """
    rets = simple_returns(prices)
    if window is not None:
        vol = rets.rolling(window).std().iloc[-1] * np.sqrt(252)
    else:
        vol = rets.std() * np.sqrt(252)
    return vol


# ---------------------------------------------------------------------------
# Data-cleaning utilities
# ---------------------------------------------------------------------------


def fix_price_anomalies(
    prices: pd.DataFrame,
    max_daily_change: float = 0.5,
    max_anomalies: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """Detect and fix assets with extreme price jumps.

    Anomalous values are replaced with the previous day's price.  Assets
    with more than *max_anomalies* anomalies are flagged for removal.

    Parameters
    ----------
    prices:
        Price DataFrame (index=date, columns=tickers).
    max_daily_change:
        Maximum acceptable absolute daily price change (default 50 %).
    max_anomalies:
        Maximum number of anomalies before flagging for removal (default 3).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(corrected_prices, assets_to_exclude)``
    """
    prices_corrected = prices.copy()
    price_changes = prices.pct_change(fill_method=None)
    total_fixes = 0
    assets_to_exclude: list[str] = []

    for col in prices.columns:
        extreme = price_changes[col][abs(price_changes[col]) > max_daily_change]
        if len(extreme) == 0:
            continue

        if len(extreme) > max_anomalies:
            assets_to_exclude.append(col)
            warnings.warn(
                f"Asset {col} has {len(extreme)} anomalies (>{max_anomalies}), will be excluded"
            )
            continue

        for date, _ in extreme.items():
            idx = prices.index.get_loc(date)
            if idx > 0:
                prev_date = prices.index[idx - 1]
                prices_corrected.loc[date, col] = prices.loc[prev_date, col]
                total_fixes += 1

    if total_fixes > 0:
        print(f"Fixed {total_fixes} anomalous price point(s)")

    if assets_to_exclude:
        prices_corrected = prices_corrected.drop(columns=assets_to_exclude)

    return prices_corrected, assets_to_exclude


def validate_and_clean_data(
    returns: pd.DataFrame,
    min_observations: int = 100,
    min_variance: float = 1e-6,
    max_nan_percentage: float = 0.2,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Remove assets with too many NaNs or near-zero variance.

    Parameters
    ----------
    returns:
        Returns DataFrame (index=date, columns=tickers).
    min_observations:
        Minimum required non-NaN rows.
    min_variance:
        Minimum acceptable variance.
    max_nan_percentage:
        Maximum fraction of NaN values before removal.

    Returns
    -------
    tuple[pd.DataFrame, list[tuple[str, str]]]
        ``(cleaned_returns, excluded_assets)`` where ``excluded_assets`` is a
        list of ``(ticker, reason)`` tuples.
    """
    excluded_assets: list[tuple[str, str]] = []
    valid_columns: list[str] = []

    for col in returns.columns:
        series = returns[col]
        nan_count = series.isna().sum()

        if nan_count > len(series) * max_nan_percentage:
            excluded_assets.append((col, f"Too many NaN values ({nan_count}/{len(series)})"))
            continue

        valid_obs = series.dropna()
        if len(valid_obs) < min_observations:
            excluded_assets.append((col, f"Insufficient observations ({len(valid_obs)} < {min_observations})"))
            continue

        variance = valid_obs.var()
        if np.isnan(variance) or variance < min_variance:
            excluded_assets.append((col, f"Zero or near-zero variance ({variance})"))
            continue

        if valid_obs.nunique() <= 1:
            excluded_assets.append((col, "Constant values"))
            continue

        valid_columns.append(col)

    if len(valid_columns) == 0:
        raise ValueError("No valid assets remaining after data validation")

    cleaned = returns[valid_columns].copy().ffill().bfill().dropna()

    if excluded_assets:
        warnings.warn(f"Excluded {len(excluded_assets)} assets due to data quality issues")

    return cleaned, excluded_assets


# ---------------------------------------------------------------------------
# Portfolio formatting helpers
# ---------------------------------------------------------------------------


def format_portfolio(
    weights: np.ndarray,
    tickers_list: list[str],
    portfolio_value: float = 100,
    to_txt: bool = False,
    txt_file_name: str | None = None,
    decimals: int = 2,
    all_assets: bool = False,
) -> str | None:
    """Build a human-readable asset allocation string.

    Parameters
    ----------
    weights:
        Weight array, one element per ticker.
    tickers_list:
        Ticker list aligned with *weights*.
    portfolio_value:
        Total portfolio value for monetary allocation (default 100).
    to_txt:
        Write the result to a ``.txt`` file when ``True``.
    txt_file_name:
        File name (without extension) for the output file.
    decimals:
        Decimal places for percentage display.
    all_assets:
        Include zero-weight assets when ``True``.

    Returns
    -------
    str | None
        Formatted allocation string, or ``None`` on invalid input.
    """
    if weights is None:
        print("Weights list is None")
        return None
    if not tickers_list:
        print("Tickers list doesn't exist")
        return None
    if weights.shape[0] != len(tickers_list):
        print("Weights list and Tickers list must have the same length")
        return None
    if to_txt and txt_file_name is None:
        txt_file_name = "weights repartition"

    lines: list[str] = []
    for i, ticker in enumerate(tickers_list):
        weight_pct = round(weights[i] * 100, decimals)
        weight_value = round(weights[i] * portfolio_value, 2)
        if all_assets or weight_value > 0:
            lines.append(
                f"{ticker}   {get_company_name(ticker)}   {weight_pct}%   {weight_value}"
            )

    repartition = "\n".join(lines) + "\n"

    if to_txt and txt_file_name:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, txt_file_name + ".txt")
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(repartition)

    return repartition


def delete_assets(excluded_assets: list, market: str) -> None:
    """Remove assets from the tickers_list.json for the given market.

    Parameters
    ----------
    excluded_assets:
        List of ticker strings or ``(column, ticker)`` tuples to remove.
    market:
        Market name key as it appears in ``tickers_list.json``.
    """
    try:
        json_path = os.path.join(os.path.dirname(__file__), "tickers_list.json")
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        tickers: list[str] = data[market]["Tickers list"]
        excluded_tickers = [
            asset[1] if isinstance(asset, tuple) else asset for asset in excluded_assets
        ]

        if check_assets_in_market(excluded_tickers, tickers):
            exclude_set = set(excluded_tickers)
            data[market]["Tickers list"] = [t for t in tickers if t not in exclude_set]
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        else:
            print("Assets to exclude are not present in this market")
    except Exception as exc:
        print(f"Error while modifying the json: {exc}")


def check_assets_in_market(assets: list[str], market_list: list[str]) -> bool:
    """Check that all asset tickers appear in *market_list*.

    Parameters
    ----------
    assets:
        Tickers to verify.
    market_list:
        Reference list of known tickers.

    Returns
    -------
    bool
        ``True`` when all assets are found.
    """
    return all(a in market_list for a in assets)


def sector_diversification(
    tickers_list: list[str],
    to_txt: bool = False,
    txt_file_name: str = "portfolio diversification",
) -> dict | None:
    """Summarise sector exposure of a portfolio.

    Parameters
    ----------
    tickers_list:
        Portfolio ticker list.
    to_txt:
        Write a summary ``.txt`` file when ``True``.
    txt_file_name:
        Output file name (without extension).

    Returns
    -------
    dict | None
        Mapping of sector name → count of assets in that sector.
    """
    if not tickers_list:
        print("No ticker found in the list")
        return None

    diversification: dict[str, int] = defaultdict(int)
    for ticker in tickers_list:
        sector = get_company_sector(ticker)
        diversification[sector] += 1

    if to_txt:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, txt_file_name + ".txt")
        result = "\n".join(f"{s}   {c}" for s, c in diversification.items()) + "\n"
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(result)

    return dict(diversification)
