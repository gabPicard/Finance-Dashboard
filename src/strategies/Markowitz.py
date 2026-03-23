"""Markowitz mean-variance optimisation strategy.

Implements both the minimum-variance and maximum-Sharpe portfolios using
``scipy.optimize.minimize`` with the SLSQP solver.

Legacy functions (``optimize_portfolio``, ``rolling_window``, etc.) are retained
for backward compatibility with pipeline code.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data.data_engineering import simple_returns
from ..metrics.portfolio_measurements import compound_growth_rate, realized_returns
from ..results.portfolio import Portfolio
from .base_allocation import AllocationStrategy


# ---------------------------------------------------------------------------
# Low-level helpers (used by both the class and legacy pipeline functions)
# ---------------------------------------------------------------------------


def _portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    """Return w^T Σ w."""
    return float(weights @ cov @ weights)


def _neg_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
) -> float:
    """Return the negative Sharpe ratio (for minimisation)."""
    ret = float(weights @ expected_returns)
    vol = float(np.sqrt(_portfolio_variance(weights, cov)))
    if vol < 1e-12:
        return 0.0
    return -(ret - risk_free_rate) / vol


def optimize_portfolio(
    cov_matrix: pd.DataFrame | np.ndarray,
    expected_returns: pd.Series | np.ndarray,
    target_return: float | None = None,
    max_weight: float = 0.15,
) -> np.ndarray | None:
    """Compute minimum-variance weights, optionally targeting a return level.

    Parameters
    ----------
    cov_matrix:
        Annualised covariance matrix (n × n).
    expected_returns:
        Annualised expected returns (length n).
    target_return:
        Optional target portfolio return constraint.
    max_weight:
        Maximum weight per asset (default 0.15).

    Returns
    -------
    np.ndarray | None
        Optimal weight vector, or ``None`` if the solver failed.
    """
    if isinstance(cov_matrix, pd.DataFrame):
        cov = cov_matrix.values.astype(float)
    else:
        cov = np.asarray(cov_matrix, dtype=float)

    if isinstance(expected_returns, pd.Series):
        mu = expected_returns.values.astype(float)
    else:
        mu = np.asarray(expected_returns, dtype=float)

    n = len(mu)
    max_w = max(min(max_weight, 1.0), 1.0 / n)

    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda w, r=target_return: w @ mu - r})

    bounds = [(0.0, max_w)] * n
    x0 = np.full(n, 1.0 / n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            _portfolio_variance,
            x0,
            args=(cov,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

    if res.success:
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    return None


def max_sharpe_portfolio(
    cov_matrix: pd.DataFrame | np.ndarray,
    expected_returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> np.ndarray | None:
    """Find the maximum-Sharpe-ratio portfolio.

    Parameters
    ----------
    cov_matrix:
        Annualised covariance matrix.
    expected_returns:
        Annualised expected returns.
    risk_free_rate:
        Risk-free rate (default 0.0).
    max_weight:
        Per-asset weight cap (default 0.15).

    Returns
    -------
    np.ndarray | None
        Optimal weight vector, or ``None`` if the solver failed.
    """
    if isinstance(cov_matrix, pd.DataFrame):
        cov = cov_matrix.values.astype(float)
    else:
        cov = np.asarray(cov_matrix, dtype=float)

    if isinstance(expected_returns, pd.Series):
        mu = expected_returns.values.astype(float)
    else:
        mu = np.asarray(expected_returns, dtype=float)

    n = len(mu)
    max_w = max(min(max_weight, 1.0), 1.0 / n)
    bounds = [(0.0, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n, 1.0 / n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            _neg_sharpe,
            x0,
            args=(mu, cov, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

    if res.success:
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    return None


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> dict[str, float]:
    """Compute portfolio return and volatility.

    Parameters
    ----------
    weights:
        Weight vector.
    expected_returns:
        Expected returns vector.
    cov_matrix:
        Covariance matrix.

    Returns
    -------
    dict[str, float]
        ``{"return": ..., "std": ...}``
    """
    port_return = float(np.dot(weights, expected_returns))
    port_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return {"return": port_return, "std": float(np.sqrt(max(port_var, 0.0)))}


def sharpe_ratio(
    portfolio_return: float, std: float, risk_free_rate: float
) -> float:
    """Local Sharpe ratio helper for use in legacy pipeline code."""
    if std == 0 or np.isnan(std):
        return 0.0
    return (portfolio_return - risk_free_rate) / std


def calculate_efficient_frontier(
    cov_matrix: pd.DataFrame | np.ndarray,
    expected_returns: pd.Series | np.ndarray,
    num_portfolios: int = 100,
    max_weight: float = 0.15,
) -> dict[str, Any]:
    """Trace the efficient frontier.

    Parameters
    ----------
    cov_matrix:
        Annualised covariance matrix.
    expected_returns:
        Annualised expected returns.
    num_portfolios:
        Number of target-return levels to solve (default 100).
    max_weight:
        Per-asset weight cap.

    Returns
    -------
    dict
        ``{"weights": np.ndarray, "returns": np.ndarray, "std": list[float]}``
    """
    if isinstance(expected_returns, pd.Series):
        mu = expected_returns.values.astype(float)
    else:
        mu = np.asarray(expected_returns, dtype=float)

    targets = np.linspace(mu.min(), mu.max(), num_portfolios)
    weights_list, std_list, valid_returns = [], [], []

    for target in targets:
        w = optimize_portfolio(cov_matrix, expected_returns, target, max_weight)
        if w is not None:
            perf = portfolio_performance(w, mu, np.asarray(cov_matrix))
            weights_list.append(w)
            std_list.append(perf["std"])
            valid_returns.append(target)

    return {
        "weights": np.array(weights_list),
        "returns": np.array(valid_returns),
        "std": std_list,
    }


def best_sharpe_ratio(efficient_frontier: dict, risk_free_rate: float) -> dict:
    """Return the portfolio with the highest Sharpe ratio on the frontier."""
    returns = efficient_frontier["returns"]
    stds = efficient_frontier["std"]
    best_sharpe = -np.inf
    best_idx = 0
    for i, (r, s) in enumerate(zip(returns, stds)):
        sr = sharpe_ratio(r, s, risk_free_rate)
        if sr > best_sharpe:
            best_sharpe = sr
            best_idx = i
    return {
        "sharpe ratio": best_sharpe,
        "weights": efficient_frontier["weights"][best_idx],
        "expected return": returns[best_idx],
        "standard deviation": stds[best_idx],
    }


# ---------------------------------------------------------------------------
# Markowitz Strategy class
# ---------------------------------------------------------------------------


class MarkowitzStrategy(AllocationStrategy):
    """Markowitz mean-variance optimisation.

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
        Pre-fetched price DataFrame (index=date, columns=tickers).
    risk_free_rate:
        Annualised risk-free rate (default 0.04).
    max_weight:
        Per-asset maximum weight (default 0.15).
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
        risk_free_rate: float = 0.04,
        max_weight: float = 0.15,
        window: int = 252,
    ) -> None:
        """Initialise with price data."""
        super().__init__(name, tickers, start, end)
        self.prices = prices
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.window = window

    def run(self) -> Portfolio:
        """Compute maximum-Sharpe weights on the full data window.

        Returns
        -------
        Portfolio
            Portfolio with ``weights`` set to the max-Sharpe allocation.
        """
        rets = simple_returns(self.prices).dropna()
        cov = rets.cov() * 252
        mu = rets.mean() * 252

        weights_arr = max_sharpe_portfolio(cov, mu, self.risk_free_rate, self.max_weight)
        if weights_arr is None:
            weights_arr = optimize_portfolio(cov, mu, None, self.max_weight)
        if weights_arr is None:
            n = len(self.tickers)
            weights_arr = np.full(n, 1.0 / n)

        weights_dict = dict(zip(rets.columns, weights_arr))
        portfolio = Portfolio(
            name=self.name,
            tickers=list(rets.columns),
            weights=weights_dict,
        )
        self._last_result = portfolio
        return portfolio

    def backtest(self) -> Portfolio:
        """Rolling-window optimisation backtest.

        Returns
        -------
        Portfolio
            Portfolio with ``weights_history`` and ``returns_history`` populated.
        """
        if "date" in self.prices.columns:
            prices_indexed = self.prices.set_index("date")
        else:
            prices_indexed = self.prices.copy()
        prices_indexed.index = pd.to_datetime(prices_indexed.index)
        prices_indexed = prices_indexed.sort_index()

        rebalance_dates = prices_indexed.resample("QE").last().index
        asset_cols = list(prices_indexed.columns)
        weights_records: list[dict] = []

        for date in rebalance_dates:
            window_prices = prices_indexed.loc[:date].tail(self.window)
            if len(window_prices) < self.window // 2:
                continue

            window_rets = simple_returns(window_prices).dropna()
            cov = window_rets.cov() * 252
            mu = window_rets.mean() * 252

            eigenvalues = np.linalg.eigvalsh(cov.values)
            if np.any(eigenvalues <= 0):
                continue

            w = max_sharpe_portfolio(cov, mu, self.risk_free_rate, self.max_weight)
            if w is None:
                w = optimize_portfolio(cov, mu, None, self.max_weight)
            if w is None:
                continue

            row: dict = {col: 0.0 for col in asset_cols}
            for col, weight in zip(window_rets.columns, w):
                row[col] = float(weight)
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


# ---------------------------------------------------------------------------
# Legacy L2-optimisation helpers (kept for pipeline backward compatibility)
# ---------------------------------------------------------------------------


def l2_algorithm(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    old_weights: np.ndarray,
    rho: float,
    gamma: float,
    max_weight: float,
) -> np.ndarray | None:
    """L2-regularised portfolio optimisation."""
    n = len(expected_returns)

    def objective(w: np.ndarray) -> float:
        variance = float(w @ cov_matrix @ w)
        l2_reg = float(rho * np.sum((w - old_weights) ** 2))
        return_pen = float(-gamma * (w @ expected_returns))
        return variance + l2_reg + return_pen

    max_w = max(min(max_weight, 1.0), 1.0 / n)
    bounds = [(0.0, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = old_weights.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

    if res.success:
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    return None


def MonteCarlo_portfolio(
    precision: int,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.04,
    strategy: str = "Lowest std",
) -> np.ndarray:
    """Monte-Carlo portfolio search."""
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values

    best_weights = np.full(len(expected_returns), 1.0 / len(expected_returns))
    best_param = 0.0 if strategy == "Best sharpe" else float("inf")

    for _ in range(precision):
        w = np.random.dirichlet(np.ones(len(expected_returns)))
        perf = portfolio_performance(w, expected_returns, np.asarray(cov_matrix))
        if strategy == "Best sharpe":
            sr = sharpe_ratio(perf["return"], perf["std"], risk_free_rate)
            if sr > best_param:
                best_param = sr
                best_weights = w
        else:
            if perf["std"] < best_param:
                best_param = perf["std"]
                best_weights = w

    return best_weights


def rolling_window(
    prices: pd.DataFrame,
    risk_free_rate: float,
    rebalance_frequency: str = "QE",
    strategy: str = "Best sharpe",
    max_weight: float = 0.15,
) -> pd.DataFrame:
    """Legacy rolling-window backtest helper."""
    prices_copy = prices.copy()
    if "date" in prices_copy.columns:
        prices_copy["date"] = pd.to_datetime(prices_copy["date"])
        prices_copy = prices_copy.set_index("date").sort_index()

    index_rebal = prices_copy.resample(rebalance_frequency).last().index
    asset_columns = list(prices_copy.columns)
    all_columns = asset_columns + ["sharpe_ratio", "expected_return", "std"]
    max_w = max(min(max_weight, 1.0), 1.0 / len(asset_columns))
    weights_backtest = pd.DataFrame(index=index_rebal, columns=all_columns, dtype=float)
    last_valid: dict | None = None

    for ind in index_rebal:
        price_tmp = prices_copy[:ind].tail(252)
        if len(price_tmp) < 252:
            continue

        rets = price_tmp.pct_change(fill_method=None).dropna()
        cov = rets.cov() * 252
        mu = rets.mean() * 252

        if np.any(np.linalg.eigvalsh(cov.values) <= 0):
            if last_valid:
                weights_backtest.loc[ind, asset_columns] = last_valid["weights"]
                weights_backtest.loc[ind, ["sharpe_ratio", "expected_return", "std"]] = [
                    last_valid["sharpe_ratio"], last_valid["expected_return"], last_valid["std"]
                ]
            continue

        if strategy == "Best sharpe":
            ef = calculate_efficient_frontier(cov, mu, max_weight=max_w)
            if len(ef["weights"]) == 0:
                continue
            best = best_sharpe_ratio(ef, risk_free_rate)
            w = best["weights"]
            sr = best["sharpe ratio"]
            er = best["expected return"]
            sd = best["standard deviation"]
        else:
            w = optimize_portfolio(cov, mu, None, max_w)
            if w is None:
                continue
            perf = portfolio_performance(w, mu.values, cov.values)
            sr = sharpe_ratio(perf["return"], perf["std"], risk_free_rate)
            er = perf["return"]
            sd = perf["std"]

        weights_backtest.loc[ind, asset_columns] = 0.0
        for i, col in enumerate(rets.columns):
            weights_backtest.loc[ind, col] = float(w[i])
        weights_backtest.loc[ind, "sharpe_ratio"] = sr
        weights_backtest.loc[ind, "expected_return"] = er
        weights_backtest.loc[ind, "std"] = sd
        last_valid = {"weights": w, "sharpe_ratio": sr, "expected_return": er, "std": sd}

    return weights_backtest


def l2_optimization(
    prices: pd.DataFrame,
    risk_free_rate: float,
    rho: float,
    gamma: float,
    rebalance_frequency: str = "QE",
    max_weight: float = 0.15,
    computation_function: str = "cgr",
) -> pd.DataFrame:
    """Legacy L2-optimisation rolling backtest."""
    prices_copy = prices.copy()
    if "date" in prices_copy.columns:
        prices_copy["date"] = pd.to_datetime(prices_copy["date"])
        prices_copy = prices_copy.set_index("date").sort_index()

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

        if computation_function == "cgr":
            mu = compound_growth_rate(price_tmp, 252)
        else:
            mu = rets.mean() * 252

        mu_arr = mu.values if isinstance(mu, pd.Series) else np.asarray(mu)
        opt_w = l2_algorithm(mu_arr, cov.values, weights_old, rho, gamma, max_w)

        if opt_w is None:
            weights_backtest.loc[ind, asset_columns] = weights_old
            weights_backtest.loc[ind, ["sharpe_ratio", "expected_return", "std"]] = [np.nan, np.nan, np.nan]
            continue

        perf = portfolio_performance(opt_w, mu_arr, cov.values)
        sr = sharpe_ratio(perf["return"], perf["std"], risk_free_rate)

        weights_backtest.loc[ind, asset_columns] = 0.0
        for i, col in enumerate(rets.columns):
            weights_backtest.loc[ind, col] = float(opt_w[i])
        weights_backtest.loc[ind, "sharpe_ratio"] = sr
        weights_backtest.loc[ind, "expected_return"] = perf["return"]
        weights_backtest.loc[ind, "std"] = perf["std"]
        weights_old = opt_w

    return weights_backtest


def find_best_params(
    prices: pd.DataFrame,
    risk_free_rate: float,
    deciding_value: str = "std",
) -> tuple[float, float]:
    """Grid-search for best L2 hyper-parameters."""
    gammas = np.linspace(0.01, 0.05, 5)
    rhos = np.linspace(0.01, 1.0, 20)
    best_gamma, best_rho = gammas[0], rhos[0]
    max_metric = -np.inf if deciding_value in ("return", "sharpe") else np.inf

    for g in gammas:
        for r in rhos:
            res = l2_optimization(prices, risk_free_rate, r, g)
            if res is None or res.empty:
                continue
            last = res[["expected_return", "sharpe_ratio", "std"]].dropna()
            if last.empty:
                continue
            row = last.iloc[-1]
            if deciding_value == "return" and row["expected_return"] > max_metric:
                max_metric = row["expected_return"]; best_gamma, best_rho = g, r
            elif deciding_value == "sharpe" and row["sharpe_ratio"] > max_metric:
                max_metric = row["sharpe_ratio"]; best_gamma, best_rho = g, r
            elif deciding_value == "std" and row["std"] < max_metric:
                max_metric = row["std"]; best_gamma, best_rho = g, r

    return best_gamma, best_rho
