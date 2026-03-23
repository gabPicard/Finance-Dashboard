"""Standalone portfolio performance measurement functions.

All functions accept a returns Series or DataFrame and return a scalar or
Series.  Dependencies are limited to numpy and pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """Compute the annualised Sharpe ratio.

    Parameters
    ----------
    returns:
        Daily portfolio returns.
    risk_free_rate:
        Annualised risk-free rate (default 0.0).
    periods:
        Trading periods per year (default 252 for daily data).

    Returns
    -------
    float
        Annualised Sharpe ratio, or 0.0 when standard deviation is zero.
    """
    if returns.empty:
        return 0.0
    daily_rfr = risk_free_rate / periods
    excess = returns - daily_rfr
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum drawdown from a returns series.

    Parameters
    ----------
    returns:
        Daily portfolio returns.

    Returns
    -------
    float
        Maximum drawdown as a non-positive fraction (e.g. -0.35 for -35 %).
    """
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def annualised_return(returns: pd.Series, periods: int = 252) -> float:
    """Compute the compounded annualised return.

    Parameters
    ----------
    returns:
        Daily portfolio returns.
    periods:
        Trading periods per year (default 252).

    Returns
    -------
    float
        Annualised return as a decimal.
    """
    if returns.empty:
        return 0.0
    total = (1 + returns).prod()
    n_periods = len(returns)
    return float(total ** (periods / n_periods) - 1)


def annualised_volatility(returns: pd.Series, periods: int = 252) -> float:
    """Compute the annualised volatility (standard deviation of returns).

    Parameters
    ----------
    returns:
        Daily portfolio returns.
    periods:
        Trading periods per year (default 252).

    Returns
    -------
    float
        Annualised volatility as a decimal.
    """
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods))


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Estimate the historical Value-at-Risk at a given confidence level.

    Parameters
    ----------
    returns:
        Daily portfolio returns.
    confidence:
        Confidence level (default 0.95 → 95 % VaR).

    Returns
    -------
    float
        VaR as a non-positive decimal (e.g. -0.02 means a 2 % daily loss at
        the specified confidence level).
    """
    if returns.empty:
        return 0.0
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Compute the portfolio beta relative to a market benchmark.

    Parameters
    ----------
    returns:
        Portfolio daily returns.
    market_returns:
        Market (benchmark) daily returns aligned to the same index.

    Returns
    -------
    float
        Beta coefficient.
    """
    aligned = pd.DataFrame({"port": returns, "mkt": market_returns}).dropna()
    if len(aligned) < 2:
        return 0.0
    mkt_var = aligned["mkt"].var(ddof=1)
    if mkt_var == 0:
        return 0.0
    return float(aligned["port"].cov(aligned["mkt"]) / mkt_var)


def alpha(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute Jensen's alpha (annualised).

    Parameters
    ----------
    returns:
        Portfolio daily returns.
    market_returns:
        Market (benchmark) daily returns.
    risk_free_rate:
        Annualised risk-free rate (default 0.0).

    Returns
    -------
    float
        Annualised Jensen's alpha as a decimal.
    """
    b = beta(returns, market_returns)
    daily_rfr = risk_free_rate / 252
    port_excess = returns - daily_rfr
    mkt_excess = market_returns - daily_rfr
    alpha_daily = port_excess.mean() - b * mkt_excess.mean()
    return float(alpha_daily * 252)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with pipelines.py)
# ---------------------------------------------------------------------------


def realized_returns(
    weights_backtest: pd.DataFrame,
    prices: pd.DataFrame,
    initial_value: float = 100,
) -> dict:
    """Calculate realised portfolio returns from backtested weights.

    Parameters
    ----------
    weights_backtest:
        DataFrame with portfolio weights at each rebalancing date.  Must
        include columns for every asset as well as ``sharpe_ratio``,
        ``expected_return``, and ``std`` columns.
    prices:
        Price DataFrame (with a ``date`` column or DatetimeIndex).
    initial_value:
        Starting portfolio value (default 100).

    Returns
    -------
    dict
        Keys: ``portfolio_value``, ``portfolio_returns``, ``cumulative_return``,
        ``annualized_return``, ``avg_sharpe``, ``avg_std``, ``final_value``,
        ``initial_value``.
    """
    prices_copy = prices.copy()
    if "date" in prices_copy.columns:
        prices_copy["date"] = pd.to_datetime(prices_copy["date"])
        prices_copy = prices_copy.set_index("date").sort_index()

    asset_columns = [
        col for col in weights_backtest.columns
        if col not in ("sharpe_ratio", "expected_return", "std")
    ]
    weights_clean = weights_backtest.dropna(how="all", subset=asset_columns)

    if len(weights_clean) == 0:
        raise ValueError("No valid rebalancing periods found in weights_backtest")

    portfolio_values = pd.Series(index=prices_copy.index, dtype=float)
    portfolio_returns_series = pd.Series(index=prices_copy.index, dtype=float)
    rebalance_dates = weights_clean.index
    current_value = initial_value

    for i in range(len(rebalance_dates)):
        target_date = rebalance_dates[i]
        next_target = (
            rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else prices_copy.index[-1]
        )

        available = prices_copy.index[prices_copy.index >= target_date]
        if len(available) == 0:
            continue
        start_date = available[0]

        available_end = prices_copy.index[prices_copy.index >= next_target]
        end_date = available_end[0] if len(available_end) > 0 else prices_copy.index[-1]

        weights_arr = weights_clean.loc[target_date, asset_columns].values.astype(float)
        valid_mask = ~np.isnan(weights_arr)
        weights_arr = weights_arr[valid_mask]
        valid_assets = np.array(asset_columns)[valid_mask]

        if weights_arr.sum() > 0:
            weights_arr = weights_arr / weights_arr.sum()
        else:
            continue

        rebalance_prices = prices_copy.loc[start_date, valid_assets]
        shares: dict[str, float] = {}
        for j, asset in enumerate(valid_assets):
            price = rebalance_prices[asset]
            if not np.isnan(price) and price > 0:
                shares[asset] = float(current_value * weights_arr[j]) / float(price)
            else:
                shares[asset] = 0.0

        period_prices = prices_copy.loc[start_date:end_date, valid_assets]
        prev_value: float | None = None

        for date in period_prices.index:
            daily_prices = period_prices.loc[date]
            pv = sum(
                shares.get(a, 0.0) * daily_prices[a]
                for a in valid_assets
                if not np.isnan(daily_prices[a])
            )
            portfolio_values[date] = pv

            if prev_value is not None and prev_value > 0:
                portfolio_returns_series[date] = (pv - prev_value) / prev_value
            else:
                portfolio_returns_series[date] = 0.0
            prev_value = pv

        current_value = float(prev_value) if prev_value is not None else current_value

    portfolio_values = portfolio_values.dropna()
    portfolio_returns_series = portfolio_returns_series.dropna()

    cumulative = (portfolio_values.iloc[-1] - initial_value) / initial_value
    n_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    ann_ret = (
        (portfolio_values.iloc[-1] / initial_value) ** (1 / n_years) - 1
        if n_years > 0
        else 0.0
    )

    avg_sharpe = weights_clean["sharpe_ratio"].astype(float).mean()
    avg_std = weights_clean["std"].astype(float).mean()

    return {
        "portfolio_value": portfolio_values,
        "portfolio_returns": portfolio_returns_series,
        "cumulative_return": cumulative,
        "annualized_return": ann_ret,
        "avg_sharpe": avg_sharpe,
        "avg_std": avg_std,
        "final_value": float(portfolio_values.iloc[-1]),
        "initial_value": initial_value,
    }


def compound_growth_rate(prices: pd.DataFrame, duration: int) -> pd.Series:
    """Compute the compound growth rate for each asset.

    Parameters
    ----------
    prices:
        Price DataFrame.
    duration:
        Annualisation period (e.g. 252 for daily data).

    Returns
    -------
    pd.Series
        CGR per ticker.
    """
    n = len(prices)
    return (prices.iloc[-1] / prices.iloc[0]) ** (duration / n) - 1
