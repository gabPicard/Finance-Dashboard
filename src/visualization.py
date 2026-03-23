"""Plotly-based visualisation functions for portfolios and trading books.

Each function returns a ``plotly.graph_objects.Figure`` that can be displayed
in a Jupyter notebook, a Dash app, or saved as an HTML file.

Legacy matplotlib helpers (``portfolio_analysis``, ``market_comparison``) are
retained for backward compatibility with existing pipeline code.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .results.portfolio import Portfolio
from .results.trading_book import TradingBook


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------


def plot_weights(portfolio: Portfolio) -> go.Figure:
    """Bar chart of the current portfolio weights.

    Parameters
    ----------
    portfolio:
        A :class:`Portfolio` instance with ``weights`` populated.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive bar chart.
    """
    if not portfolio.weights:
        return go.Figure().update_layout(title="No weights available")

    tickers = list(portfolio.weights.keys())
    values = [portfolio.weights[t] * 100 for t in tickers]

    fig = go.Figure(
        go.Bar(
            x=tickers,
            y=values,
            marker_color="steelblue",
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"{portfolio.name} — Portfolio Weights",
        xaxis_title="Ticker",
        yaxis_title="Weight (%)",
        yaxis=dict(ticksuffix="%"),
        template="plotly_white",
    )
    return fig


def plot_returns(portfolio: Portfolio) -> go.Figure:
    """Line chart of cumulative returns from ``returns_history``.

    Parameters
    ----------
    portfolio:
        A :class:`Portfolio` instance with ``returns_history`` populated.

    Returns
    -------
    plotly.graph_objects.Figure
        Cumulative return chart.
    """
    if portfolio.returns_history.empty:
        return go.Figure().update_layout(title="No returns history available")

    cumulative = (1 + portfolio.returns_history).cumprod() - 1

    fig = go.Figure(
        go.Scatter(
            x=cumulative.index,
            y=cumulative.values * 100,
            mode="lines",
            name="Cumulative Return",
            line=dict(color="green", width=2),
        )
    )
    fig.update_layout(
        title=f"{portfolio.name} — Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        yaxis=dict(ticksuffix="%"),
        template="plotly_white",
    )
    return fig


def plot_drawdown(portfolio: Portfolio) -> go.Figure:
    """Area chart of the drawdown series.

    Parameters
    ----------
    portfolio:
        A :class:`Portfolio` instance with ``returns_history`` populated.

    Returns
    -------
    plotly.graph_objects.Figure
        Drawdown chart.
    """
    if portfolio.returns_history.empty:
        return go.Figure().update_layout(title="No returns history available")

    cumulative = (1 + portfolio.returns_history).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max * 100

    fig = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.2)",
            line=dict(color="red", width=1.5),
            name="Drawdown",
        )
    )
    fig.update_layout(
        title=f"{portfolio.name} — Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis=dict(ticksuffix="%"),
        template="plotly_white",
    )
    return fig


def plot_pair_spread(trading_book: TradingBook) -> go.Figure:
    """Multi-trace chart of pair spreads from ``positions_history``.

    Parameters
    ----------
    trading_book:
        A :class:`TradingBook` with ``positions_history`` populated.

    Returns
    -------
    plotly.graph_objects.Figure
        Spread chart with one trace per ticker position.
    """
    if trading_book.positions_history.empty:
        return go.Figure().update_layout(title="No positions history available")

    fig = go.Figure()
    for ticker in trading_book.positions_history.columns:
        fig.add_trace(
            go.Scatter(
                x=trading_book.positions_history.index,
                y=trading_book.positions_history[ticker],
                mode="lines",
                name=ticker,
            )
        )
    fig.update_layout(
        title=f"{trading_book.name} — Pair Spread Positions",
        xaxis_title="Date",
        yaxis_title="Position Signal",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig


def plot_pnl(trading_book: TradingBook) -> go.Figure:
    """Cumulative P&L chart from ``pnl_history``.

    Parameters
    ----------
    trading_book:
        A :class:`TradingBook` with ``pnl_history`` populated.

    Returns
    -------
    plotly.graph_objects.Figure
        Cumulative P&L chart.
    """
    if trading_book.pnl_history.empty:
        return go.Figure().update_layout(title="No P&L history available")

    cum_pnl = trading_book.pnl_history.cumsum()

    fig = go.Figure(
        go.Scatter(
            x=cum_pnl.index,
            y=cum_pnl.values,
            mode="lines",
            line=dict(color="purple", width=2),
            name="Cumulative P&L",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=f"{trading_book.name} — Cumulative P&L",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Legacy matplotlib helpers (backward compatibility)
# ---------------------------------------------------------------------------


def portfolio_analysis(data: pd.DataFrame, title: str, prices: pd.DataFrame | None = None,
                       realized_returns: dict | None = None) -> None:
    """Legacy matplotlib portfolio visualisation (kept for pipeline compatibility)."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.4)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax_stats = fig.add_subplot(gs[:, 2])

    asset_cols = [c for c in data.columns if c not in ("sharpe_ratio", "expected_return", "std")]
    data[asset_cols].abs().plot(kind="area", stacked=True, cmap="tab20", ax=ax1)
    ax1.set_ylabel("Weight"); ax1.set_title("Asset Weights Over Time", fontweight="bold")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    ax2.plot(data.index, data["sharpe_ratio"], color="green")
    ax2.set_title("Sharpe Ratio Over Time"); ax2.grid(True, alpha=0.3)

    ax3.plot(data.index, data["expected_return"], color="blue")
    ax3.set_title("Expected Return Over Time"); ax3.grid(True, alpha=0.3)

    ax4.plot(data.index, data["std"], color="red")
    ax4.set_title("Standard Deviation Over Time"); ax4.grid(True, alpha=0.3)

    if realized_returns:
        pv = realized_returns["portfolio_value"]
        iv = realized_returns["initial_value"]
        ax5.plot(pv.index, pv.values, color="purple")
        ax5.axhline(y=iv, color="grey", linestyle="--")
        ax5.set_title("Portfolio Value Over Time"); ax5.grid(True, alpha=0.3)

        stats = (
            f"Total Return: {realized_returns['cumulative_return']:.2%}\n"
            f"Ann. Return: {realized_returns['annualized_return']:.2%}\n"
            f"Avg Sharpe: {realized_returns['avg_sharpe']:.3f}\n"
            f"Avg Std: {realized_returns['avg_std']:.2%}"
        )
        ax_stats.axis("off")
        ax_stats.text(0.05, 0.95, stats, transform=ax_stats.transAxes, fontsize=10,
                      verticalalignment="top", fontfamily="monospace",
                      bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    plt.show()


def market_comparison(portfolio_perf: dict, market_prices: pd.DataFrame,
                      initial_value: float = 100, title: str = "Portfolio vs Market") -> None:
    """Legacy matplotlib market comparison chart."""
    if portfolio_perf is None or "portfolio_value" not in portfolio_perf:
        print("Error: provide portfolio_perf dict with 'portfolio_value' key")
        return

    pv = portfolio_perf["portfolio_value"]
    if isinstance(market_prices, pd.DataFrame):
        mkt = market_prices.iloc[:, 0]
    else:
        mkt = market_prices

    mkt = mkt[(mkt.index >= pv.index[0]) & (mkt.index <= pv.index[-1])]
    mkt_norm = (mkt / mkt.iloc[0]) * initial_value

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(pv.index, pv.values, color="#2E86AB", label="Portfolio", linewidth=2.5)
    ax.plot(mkt_norm.index, mkt_norm.values, color="#F18F01", label="Market Index", linewidth=2.5)
    ax.axhline(y=initial_value, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Date"); ax.set_ylabel("Value ($)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
