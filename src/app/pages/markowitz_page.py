"""Markowitz optimisation page for the Finance Dashboard Dash app."""

from __future__ import annotations

import json

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from ...data.stock_prices import get_tickers_list
from ...strategies.Markowitz import (
    MarkowitzStrategy,
    calculate_efficient_frontier,
    max_sharpe_portfolio,
    optimize_portfolio,
    portfolio_performance,
)
from ...visualization import plot_drawdown, plot_returns, plot_weights

dash.register_page(__name__, path="/markowitz", name="Markowitz Optimiser")


# ---------------------------------------------------------------------------
# Helper: list available markets from tickers_list.json
# ---------------------------------------------------------------------------
def _get_markets() -> list[str]:
    import os, json as _json
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tickers_list.json")
    try:
        with open(path) as fh:
            return list(_json.load(fh).keys())
    except Exception:
        return ["S&P500", "CAC40"]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto"},
    children=[
        html.H2("Markowitz Portfolio Optimiser", style={"color": "#2c3e50"}),
        html.P(
            "Select a market and period, then click Optimise to compute the "
            "maximum-Sharpe portfolio.",
            style={"color": "#7f8c8d"},
        ),
        html.Hr(),

        # ── Controls ────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
            children=[
                html.Div(
                    children=[
                        html.Label("Market", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="mw-market-dropdown",
                            options=[{"label": m, "value": m} for m in _get_markets()],
                            value="S&P500",
                            clearable=False,
                            style={"minWidth": "180px"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Period", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="mw-period-dropdown",
                            options=[
                                {"label": "1 Year", "value": "1y"},
                                {"label": "2 Years", "value": "2y"},
                                {"label": "3 Years", "value": "3y"},
                                {"label": "5 Years", "value": "5y"},
                            ],
                            value="2y",
                            clearable=False,
                            style={"minWidth": "140px"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Max Weight (%)", style={"fontWeight": "bold"}),
                        dcc.Slider(
                            id="mw-max-weight-slider",
                            min=5, max=50, step=5, value=15,
                            marks={v: f"{v}%" for v in range(5, 55, 10)},
                            tooltip={"placement": "bottom"},
                        ),
                    ],
                    style={"minWidth": "300px"},
                ),
                html.Div(
                    children=[
                        html.Label("\u00a0"),
                        html.Button(
                            "Optimise",
                            id="mw-run-button",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#2c3e50",
                                "color": "white",
                                "border": "none",
                                "padding": "8px 20px",
                                "borderRadius": "4px",
                                "cursor": "pointer",
                                "fontSize": "14px",
                            },
                        ),
                    ]
                ),
            ],
        ),

        # ── Status message ───────────────────────────────────────────────
        html.Div(id="mw-status", style={"color": "#e74c3c", "marginBottom": "8px"}),

        # ── Result panels ────────────────────────────────────────────────
        html.Div(
            id="mw-results-container",
            children=[
                dcc.Loading(
                    id="mw-loading",
                    type="circle",
                    children=[
                        html.Div(id="mw-metrics-panel", style={"marginBottom": "16px"}),
                        dcc.Graph(id="mw-weights-chart"),
                        dcc.Graph(id="mw-efficient-frontier"),
                    ],
                )
            ],
        ),

        # ── Hidden store ─────────────────────────────────────────────────
        dcc.Store(id="mw-portfolio-store"),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("mw-weights-chart", "figure"),
    Output("mw-efficient-frontier", "figure"),
    Output("mw-metrics-panel", "children"),
    Output("mw-status", "children"),
    Input("mw-run-button", "n_clicks"),
    State("mw-market-dropdown", "value"),
    State("mw-period-dropdown", "value"),
    State("mw-max-weight-slider", "value"),
    prevent_initial_call=True,
)
def run_optimisation(n_clicks: int, market: str, period: str, max_weight_pct: int):
    """Fetch data, optimise, and update all charts."""
    import plotly.graph_objects as go

    empty_fig = go.Figure().update_layout(template="plotly_white")
    max_w = max_weight_pct / 100.0

    try:
        from ...data.stock_prices import get_stock_prices
        prices_df, _, rfr = get_stock_prices(market, period=period)

        if "date" in prices_df.columns:
            prices_indexed = prices_df.set_index("date")
        else:
            prices_indexed = prices_df.copy()
        prices_indexed.index = pd.to_datetime(prices_indexed.index)

        rets = prices_indexed.pct_change(fill_method=None).dropna()
        cov = rets.cov() * 252
        mu = rets.mean() * 252

        weights_arr = max_sharpe_portfolio(cov, mu, rfr, max_w)
        if weights_arr is None:
            weights_arr = optimize_portfolio(cov, mu, None, max_w)
        if weights_arr is None:
            return empty_fig, empty_fig, "", "Optimisation failed — try a different period or market."

        tickers = list(rets.columns)
        portfolio = __import__(
            "src.results.portfolio", fromlist=["Portfolio"]
        ).Portfolio(
            name=f"Markowitz ({market})",
            tickers=tickers,
            weights=dict(zip(tickers, weights_arr)),
        )

        weights_fig = plot_weights(portfolio)

        # Efficient frontier
        ef = calculate_efficient_frontier(cov, mu, num_portfolios=80, max_weight=max_w)
        if len(ef["weights"]) > 0:
            ef_fig = go.Figure()
            ef_fig.add_trace(go.Scatter(
                x=[s * 100 for s in ef["std"]],
                y=[r * 100 for r in ef["returns"]],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color="royalblue"),
                name="Efficient Frontier",
            ))
            perf = portfolio_performance(weights_arr, mu.values, cov.values)
            ef_fig.add_trace(go.Scatter(
                x=[perf["std"] * 100],
                y=[perf["return"] * 100],
                mode="markers",
                marker=dict(size=12, color="red", symbol="star"),
                name="Max Sharpe",
            ))
            ef_fig.update_layout(
                title="Efficient Frontier",
                xaxis_title="Volatility (%)",
                yaxis_title="Expected Return (%)",
                template="plotly_white",
            )
        else:
            ef_fig = empty_fig

        # Metrics panel
        perf = portfolio_performance(weights_arr, mu.values, cov.values)
        sharpe = (perf["return"] - rfr) / perf["std"] if perf["std"] > 0 else 0.0
        metrics_panel = html.Div(
            style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
            children=[
                _metric_card("Expected Return", f"{perf['return']*100:.2f}%", "#27ae60"),
                _metric_card("Volatility", f"{perf['std']*100:.2f}%", "#e74c3c"),
                _metric_card("Sharpe Ratio", f"{sharpe:.3f}", "#2980b9"),
                _metric_card("Risk-Free Rate", f"{rfr*100:.2f}%", "#8e44ad"),
                _metric_card("Assets", str(len([w for w in weights_arr if w > 0.001])), "#e67e22"),
            ],
        )

        return weights_fig, ef_fig, metrics_panel, ""

    except Exception as exc:
        return empty_fig, empty_fig, "", f"Error: {exc}"


def _metric_card(label: str, value: str, color: str) -> html.Div:
    """Small metric display card."""
    return html.Div(
        style={
            "backgroundColor": "white",
            "border": f"2px solid {color}",
            "borderRadius": "8px",
            "padding": "12px 20px",
            "textAlign": "center",
            "minWidth": "120px",
        },
        children=[
            html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
            html.Div(label, style={"fontSize": "12px", "color": "#7f8c8d", "marginTop": "4px"}),
        ],
    )
