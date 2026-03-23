"""Finance Dashboard — Dash application entry-point.

Run with:
    python -m src.app.Home

Then open http://localhost:8050 in your browser.
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path when run as a module
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "Finance Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f5f6fa"},
    children=[
        # ── Navigation bar ──────────────────────────────────────────────
        html.Nav(
            style={
                "backgroundColor": "#2c3e50",
                "padding": "12px 24px",
                "display": "flex",
                "alignItems": "center",
                "gap": "24px",
            },
            children=[
                html.Span(
                    "📈 Finance Dashboard",
                    style={"color": "white", "fontSize": "20px", "fontWeight": "bold"},
                ),
                dcc.Link(
                    "Home",
                    href="/",
                    style={"color": "#ecf0f1", "textDecoration": "none", "fontSize": "14px"},
                ),
                dcc.Link(
                    "Markowitz",
                    href="/markowitz",
                    style={"color": "#ecf0f1", "textDecoration": "none", "fontSize": "14px"},
                ),
            ],
        ),
        # ── Page content ────────────────────────────────────────────────
        html.Div(
            style={"padding": "24px"},
            children=[dash.page_container],
        ),
    ],
)


# ── Home page (inline) ───────────────────────────────────────────────────────
dash.register_page(
    "home",
    path="/",
    layout=html.Div(
        children=[
            html.H1("Welcome to Finance Dashboard", style={"color": "#2c3e50"}),
            html.P(
                "A quantitative finance framework with multi-strategy portfolio "
                "optimisation and backtesting.",
                style={"color": "#7f8c8d", "fontSize": "16px"},
            ),
            html.Hr(),
            html.H3("Available Strategies", style={"color": "#2c3e50"}),
            html.Ul(
                children=[
                    html.Li("Markowitz — Mean-variance optimisation (min-variance / max-Sharpe)"),
                    html.Li("CAPM — Capital Asset Pricing Model weighted allocation"),
                    html.Li("HRP — Hierarchical Risk Parity"),
                    html.Li("PairTrading — Cointegration-based long/short pairs"),
                ],
                style={"lineHeight": "2", "fontSize": "15px"},
            ),
            html.Hr(),
            html.P(
                "Use the navigation bar above to explore the Markowitz optimiser page.",
                style={"color": "#7f8c8d"},
            ),
        ]
    ),
)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
