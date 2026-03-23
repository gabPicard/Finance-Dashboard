"""High-level pipeline entry-points.

Provides :func:`run_pipeline` (live/paper mode) and :func:`run_backtest`
(historical simulation).  Strategies are loaded from the registry and data
is fetched via the three-layer fetch stack.

Healthcheck URL (HEALTHCHECK_URL env var) is pinged after each successful
pipeline run; a missing URL is silently logged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
import warnings
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ping_healthcheck() -> None:
    """Send a GET request to HEALTHCHECK_URL if configured."""
    url = os.getenv("HEALTHCHECK_URL", "").strip()
    if not url:
        logger.debug("HEALTHCHECK_URL not set — skipping healthcheck ping.")
        return
    try:
        requests.get(url, timeout=5)
        logger.info("Healthcheck ping sent to %s", url)
    except Exception as exc:
        logger.warning("Healthcheck ping failed: %s", exc)


def _paper_execute(strategy_name: str, result: Any) -> None:
    """Log a paper-mode 'execution' (no real orders placed).

    Parameters
    ----------
    strategy_name:
        Name of the strategy.
    result:
        The Portfolio or TradingBook returned by the strategy.
    """
    logger.info(
        "[PAPER] Strategy=%s | type=%s",
        strategy_name,
        type(result).__name__,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    strategy_names: list[str],
    mode: str = "paper",
    prices: pd.DataFrame | None = None,
    market_prices: pd.DataFrame | None = None,
    risk_free_rate: float = 0.04,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
) -> dict[str, Any]:
    """Run one or more strategies and optionally execute in paper mode.

    Parameters
    ----------
    strategy_names:
        List of strategy names as registered in :class:`src.registry.StrategyRegistry`.
    mode:
        Execution mode.  ``"paper"`` logs orders; no live mode is implemented.
    prices:
        Pre-fetched price DataFrame.  When ``None``, strategies cannot be
        instantiated without prices (caller must provide prices).
    market_prices:
        Market index price DataFrame (used by CAPM).
    risk_free_rate:
        Annualised risk-free rate (default 0.04).
    start:
        ISO-8601 start date for the data window.
    end:
        ISO-8601 end date for the data window.

    Returns
    -------
    dict[str, Any]
        Mapping of strategy name → result dict with keys ``result`` and
        ``metrics``.
    """
    from .registry import registry
    from .metrics.portfolio_measurements import (
        sharpe_ratio as _sharpe,
        max_drawdown as _mdd,
        annualised_return as _ann_ret,
    )

    tickers: list[str] = []
    if prices is not None:
        tickers = [c for c in prices.columns if c != "date"]

    results: dict[str, Any] = {}

    for name in strategy_names:
        logger.info("Running strategy: %s (mode=%s)", name, mode)
        try:
            strategy_cls = registry.get(name)
        except KeyError as exc:
            logger.error("Strategy not found: %s", exc)
            continue

        try:
            # Instantiate strategy with available data
            kwargs: dict = dict(
                name=name,
                tickers=tickers,
                start=start,
                end=end,
                prices=prices if prices is not None else pd.DataFrame(),
            )
            if name == "CAPM" and market_prices is not None:
                kwargs["market_prices"] = market_prices
                kwargs["risk_free_rate"] = risk_free_rate

            strategy = strategy_cls(**kwargs)
            result = strategy.run()
            metrics = strategy.get_metrics()

            if mode == "paper":
                _paper_execute(name, result)

            results[name] = {"result": result, "metrics": metrics}
            logger.info(
                "Strategy %s completed | metrics=%s",
                name,
                {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            )

        except Exception as exc:
            logger.error("Strategy %s failed: %s", name, exc, exc_info=True)
            results[name] = {"result": None, "metrics": {}, "error": str(exc)}

    _ping_healthcheck()
    return results


def run_backtest(
    strategy_name: str,
    tickers: list[str],
    start: str,
    end: str,
    prices: pd.DataFrame | None = None,
    market_prices: pd.DataFrame | None = None,
    risk_free_rate: float = 0.04,
) -> Any:
    """Run a single strategy in backtest mode.

    Parameters
    ----------
    strategy_name:
        Name of the strategy as registered in the registry.
    tickers:
        Asset tickers.
    start:
        ISO-8601 start date.
    end:
        ISO-8601 end date.
    prices:
        Price DataFrame (index=date, columns=tickers).
    market_prices:
        Market index price DataFrame (used by CAPM).
    risk_free_rate:
        Annualised risk-free rate (default 0.04).

    Returns
    -------
    Portfolio | TradingBook | None
        Backtest result, or ``None`` on failure.
    """
    from .registry import registry

    try:
        strategy_cls = registry.get(strategy_name)
    except KeyError as exc:
        logger.error("Strategy not found: %s", exc)
        return None

    try:
        kwargs: dict = dict(
            name=strategy_name,
            tickers=tickers,
            start=start,
            end=end,
            prices=prices if prices is not None else pd.DataFrame(),
        )
        if strategy_name == "CAPM" and market_prices is not None:
            kwargs["market_prices"] = market_prices
            kwargs["risk_free_rate"] = risk_free_rate

        strategy = strategy_cls(**kwargs)
        result = strategy.backtest()
        logger.info("Backtest complete for %s", strategy_name)
        return result

    except Exception as exc:
        logger.error("Backtest %s failed: %s", strategy_name, exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Legacy pipeline functions (backward compatibility)
# ---------------------------------------------------------------------------

def portfolio_report(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    portfolio_value: float = 100,
) -> None:
    """Create a full portfolio report using legacy visualization helpers.

    Parameters
    ----------
    weights:
        Backtest weights DataFrame (from rolling_window / l2_optimization).
    prices:
        Asset price DataFrame.
    portfolio_value:
        Monetary portfolio value for weight formatting (default 100).
    """
    from .metrics.portfolio_measurements import realized_returns
    from .data.data_engineering import format_portfolio, sector_diversification
    from .visualization import plot_weights, plot_returns

    rr = realized_returns(weights, prices)
    actual_tickers = [c for c in prices.columns if c != "date"]
    metrics = ["sharpe_ratio", "expected_return", "std"]
    pure_weights = weights[[c for c in weights.columns if c not in metrics]]
    last_weights = pure_weights.iloc[-1]
    last_weights = last_weights[last_weights.abs() > 1e-6]
    actual_tickers = list(last_weights.index)

    format_portfolio(last_weights.values, actual_tickers, portfolio_value=portfolio_value, to_txt=True)
    sector_diversification(actual_tickers, to_txt=True)

    logger.info("Portfolio report generated. Tickers: %s", actual_tickers)


def l2_capm(
    markets: str | list[str],
    rho: float,
    gamma: float,
    rebalance_frequency: str,
    max_weight: float,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """L2-CAPM rolling backtest (legacy helper).

    See :func:`src.strategies.Markowitz.l2_optimization` and
    :func:`src.strategies.CAPM.capm_expected_returns` for the underlying
    implementation details.

    Parameters
    ----------
    markets:
        Market name or list of market names.
    rho:
        L2 proximity regularisation weight.
    gamma:
        Return-maximisation coefficient.
    rebalance_frequency:
        pandas offset string (e.g. ``"QE"``).
    max_weight:
        Per-asset maximum weight.
    start_date:
        Optional start date.
    end_date:
        Optional end date.
    period:
        yfinance period (used when *start_date* is ``None``).
    interval:
        yfinance interval.

    Returns
    -------
    pd.DataFrame
        Weights DataFrame with a row per rebalancing date.
    """
    from .data.stock_prices import get_stock_prices, get_tickers_list
    from .strategies.Markowitz import l2_algorithm, portfolio_performance, sharpe_ratio
    from .strategies.CAPM import capm_expected_returns
    from .metrics.portfolio_measurements import compound_growth_rate

    if isinstance(markets, str):
        prices, market_prices, rfr = get_stock_prices(
            markets, start_date=start_date, end_date=end_date, period=period, interval=interval
        )
        prices_copy = prices.copy()
        if "date" in prices_copy.columns:
            prices_copy["date"] = pd.to_datetime(prices_copy["date"])
            prices_copy = prices_copy.set_index("date").sort_index()
        market_copy = market_prices.copy()
        if "date" in market_copy.columns:
            market_copy["date"] = pd.to_datetime(market_copy["date"])
            market_copy = market_copy.set_index("date").sort_index()
    else:
        all_prices = []
        all_market = {}
        rfr = None
        for market in markets:
            p, mp, r = get_stock_prices(
                market, start_date=start_date, end_date=end_date, period=period, interval=interval
            )
            p_tmp = p.copy()
            if "date" in p_tmp.columns:
                p_tmp["date"] = pd.to_datetime(p_tmp["date"])
                p_tmp = p_tmp.set_index("date").sort_index()
            all_prices.append(p_tmp)
            mp_tmp = mp.copy()
            if "date" in mp_tmp.columns:
                mp_tmp["date"] = pd.to_datetime(mp_tmp["date"])
                mp_tmp = mp_tmp.set_index("date").sort_index()
            all_market[market] = mp_tmp
            if rfr is None:
                rfr = r
        prices_copy = pd.concat(all_prices, axis=1).loc[:, ~pd.concat(all_prices, axis=1).columns.duplicated()]

    index_rebal = prices_copy.resample(rebalance_frequency).last().index
    asset_columns = list(prices_copy.columns)
    all_columns = asset_columns + ["sharpe_ratio", "expected_return", "std"]
    max_w = max(min(max_weight, 1.0), 1.0 / len(asset_columns))
    weights_backtest = pd.DataFrame(index=index_rebal, columns=all_columns, dtype=float)
    weights_old = np.full(len(asset_columns), 1.0 / len(asset_columns))

    for ind in index_rebal:
        price_tmp = prices_copy[:ind].tail(252)
        if len(price_tmp) < 63:
            continue
        rets = price_tmp.pct_change(fill_method=None).dropna()
        cov = rets.cov() * 252

        if isinstance(markets, str):
            mkt_tmp = market_copy[:ind].tail(252)
            exp_ret = capm_expected_returns(price_tmp, mkt_tmp, rfr)
        else:
            exp_ret = {}
            for market in markets:
                tickers_list, _, _ = get_tickers_list(market)
                mkt_assets = [c for c in price_tmp.columns if c in tickers_list]
                if not mkt_assets:
                    continue
                mkt_tmp = all_market[market][:ind].tail(252)
                er = capm_expected_returns(price_tmp[mkt_assets], mkt_tmp, rfr)
                exp_ret.update(er)

        available = [c for c in rets.columns if c in exp_ret]
        if not available:
            continue

        rets = rets[available]
        cov = rets.cov() * 252
        mu_series = pd.Series({a: exp_ret[a] for a in available})
        mu_arr = mu_series.values

        w_old_filtered = np.full(len(available), 1.0 / len(available))
        opt_w = l2_algorithm(mu_arr, cov.values, w_old_filtered, rho, gamma, max_w)

        if opt_w is None:
            weights_backtest.loc[ind, asset_columns] = weights_old
            continue

        perf = portfolio_performance(opt_w, mu_arr, cov.values)
        sr = sharpe_ratio(perf["return"], perf["std"], rfr)

        weights_backtest.loc[ind, asset_columns] = 0.0
        for i, a in enumerate(available):
            weights_backtest.loc[ind, a] = float(opt_w[i])
        weights_backtest.loc[ind, "sharpe_ratio"] = sr
        weights_backtest.loc[ind, "expected_return"] = perf["return"]
        weights_backtest.loc[ind, "std"] = perf["std"]
        weights_old = opt_w

    return weights_backtest
