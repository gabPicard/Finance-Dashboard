/**
 * bindings.cpp
 *
 * pybind11 bindings exposing the C++ finance core as a ``_core`` Python module.
 *
 * All functions accept and return numpy arrays (via pybind11/eigen.h).
 *
 * Python usage:
 *   import _core
 *   weights = _core.optimize_weights(returns_matrix, risk_aversion=0.5)
 *   port_rets = _core.run_backtest(prices, weights, rebalance_days)
 *   zscores  = _core.rolling_zscore(spread, window=30)
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backtest_engine.h"
#include "optimizer.h"
#include "spread_calc.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Finance Dashboard C++ core: Markowitz optimizer, backtest engine, "
              "and rolling z-score calculator.";

    // ── optimizer ──────────────────────────────────────────────────────────
    m.def(
        "optimize_weights",
        &finance::optimize_weights,
        py::arg("returns_matrix"),
        py::arg("risk_aversion") = 0.0,
        R"pbdoc(
Compute minimum-variance (or mean-variance) portfolio weights.

Parameters
----------
returns_matrix : numpy.ndarray, shape (T, N)
    Matrix of asset returns (rows = time steps, columns = assets).
risk_aversion : float, optional
    Risk-aversion coefficient gamma (default 0.0 = pure min-variance).
    Higher values tilt the portfolio towards higher expected return.

Returns
-------
numpy.ndarray, shape (N,)
    Optimal weight vector summing to 1, with non-negative entries.
        )pbdoc");

    // ── backtest engine ────────────────────────────────────────────────────
    m.def(
        "run_backtest",
        &finance::run_backtest,
        py::arg("price_matrix"),
        py::arg("weight_matrix"),
        py::arg("rebalance_days"),
        R"pbdoc(
Run a bar-by-bar portfolio backtest.

Parameters
----------
price_matrix : numpy.ndarray, shape (T, N)
    Asset prices (rows = days, columns = assets).
weight_matrix : numpy.ndarray, shape (K, N)
    Portfolio weights at each rebalancing date.
rebalance_days : list[int]
    Sorted row indices (into price_matrix) where rebalancing occurs.
    Length must equal K.

Returns
-------
numpy.ndarray, shape (T,)
    Daily portfolio returns.  Index 0 is always 0.0 (no previous price).
        )pbdoc");

    // ── spread z-score ─────────────────────────────────────────────────────
    m.def(
        "rolling_zscore",
        &finance::rolling_zscore,
        py::arg("spread"),
        py::arg("window"),
        R"pbdoc(
Compute the rolling z-score of a spread series.

Parameters
----------
spread : numpy.ndarray, shape (T,)
    Input spread time series.
window : int
    Rolling window size (must be >= 2).

Returns
-------
numpy.ndarray, shape (T,)
    Z-score series.  The first (window - 1) entries are NaN.
        )pbdoc");
}
