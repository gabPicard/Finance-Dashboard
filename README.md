# Finance Dashboard

A production-quality quantitative finance framework for multi-strategy portfolio
optimisation, backtesting, and live/paper trading management.

---

## Architecture

```
Finance-Dashboard/
├── data/                        # Ticker lists, portfolio snapshots
├── src/
│   ├── app/                     # Dash web application
│   │   ├── Home.py              # Entry-point (python -m src.app.Home)
│   │   └── pages/
│   │       └── markowitz_page.py
│   ├── core/                    # C++ extension module (_core)
│   │   ├── optimizer.{h,cpp}    # Markowitz QP solver (Eigen)
│   │   ├── backtest_engine.{h,cpp}  # Bar-by-bar backtest loop
│   │   ├── spread_calc.{h,cpp}  # Rolling z-score
│   │   └── bindings.cpp         # pybind11 Python bindings
│   ├── data/                    # Data fetch / cache / engineering
│   │   ├── cache.py             # Thread-safe TTL cache
│   │   ├── fetch_data.py        # 3-layer lookup (cache → DB → API)
│   │   ├── data_engineering.py  # Returns, covariance, cleaning
│   │   └── stock_prices.py      # StockPrices convenience class
│   ├── db/                      # SQLAlchemy / SQLite persistence
│   │   ├── models.py            # ORM models
│   │   ├── database.py          # Engine, session, CRUD helpers
│   │   ├── schema.sql           # Raw DDL (mirrors ORM exactly)
│   │   └── seed.py              # Idempotent data seeder
│   ├── metrics/
│   │   └── portfolio_measurements.py  # Sharpe, drawdown, VaR, alpha/beta
│   ├── results/
│   │   ├── portfolio.py         # Portfolio dataclass
│   │   └── trading_book.py      # TradingBook dataclass
│   ├── strategies/
│   │   ├── base_allocation.py   # Abstract AllocationStrategy
│   │   ├── base_trading.py      # Abstract TradingStrategy
│   │   ├── Markowitz.py         # Mean-variance optimisation
│   │   ├── CAPM.py              # Capital Asset Pricing Model
│   │   ├── HRP.py               # Hierarchical Risk Parity
│   │   └── PairTrading.py       # Cointegration pair trading
│   ├── registry.py              # StrategyRegistry + @register decorator
│   ├── pipelines.py             # run_pipeline / run_backtest entry-points
│   └── visualization.py        # Plotly charts + legacy matplotlib helpers
├── tests/                       # pytest test suite
│   ├── conftest.py              # Shared fixtures
│   ├── data/                    # cache, fetch, data_engineering tests
│   ├── db/                      # models, database, seed tests
│   ├── metrics/                 # portfolio_measurements tests
│   ├── strategies/              # Markowitz, CAPM, HRP, PairTrading tests
│   ├── results/                 # Portfolio, TradingBook tests
│   └── integration/             # Pipeline + data-stack integration tests
├── .env.example                 # Environment variable template
├── CMakeLists.txt               # C++ build configuration
├── pyproject.toml               # scikit-build-core + project metadata
└── requirements.txt             # Python dependencies
```

### Data-flow

```
yfinance / Alpaca API
        │
        ▼
   fetch_data.py   ◄──────────  PriceCache (15-min TTL)
        │                              ▲
        ▼                              │
   SQLite DB  ──── insert_prices ──────┘
        │
        ▼
 data_engineering.py  (returns, cov, vol, cleaning)
        │
        ▼
   AllocationStrategy / TradingStrategy
        │                       │
        ▼                       ▼
    Portfolio             TradingBook
        │                       │
        └──────── pipelines.py ─┘
                      │
                      ▼
              visualization.py  →  Dash App
```

---

## Setup

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

Key variables:

| Variable          | Default            | Description                              |
|-------------------|--------------------|------------------------------------------|
| `DB_PATH`         | `data/finance.db`  | Path to the SQLite database file         |
| `HEALTHCHECK_URL` | *(empty)*          | URL to ping after each pipeline run      |
| `LOG_LEVEL`       | `INFO`             | Python logging level                     |

### 3. Seed the database

```bash
python -m src.db.seed
```

### 4. Build the C++ extension (optional)

Requires CMake ≥ 3.15, a C++17 compiler, and Eigen3.

```bash
pip install scikit-build-core pybind11
pip install -e .
# or:
cmake -B build && cmake --build build --config Release
```

### 5. Run the web app

```bash
python -m src.app.Home
```

Then open [http://localhost:8050](http://localhost:8050).

---

## Running tests

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## How to add a new strategy

1. **Create** `src/strategies/MyStrategy.py`.
2. **Subclass** `AllocationStrategy` (for portfolios) or `TradingStrategy` (for
   long/short books).
3. **Implement** `run() -> Portfolio` and `backtest() -> Portfolio`.
4. **Register** it in `src/registry.py`:

```python
from .strategies.MyStrategy import MyStrategy
registry.register("MyStrategy", MyStrategy)
```

5. **Add tests** in `tests/strategies/test_my_strategy.py`.

---

## Key design decisions

- **Three-layer data lookup**: cache → SQLite → API stub.  Swap out the API stub
  in `src/data/fetch_data.py` Layer 3 block for your real data provider.
- **No circular imports**: strategies never import from pipelines; data layer never
  imports from strategies.
- **Dataclasses for results**: `Portfolio` and `TradingBook` are typed dataclasses
  with validation methods.
- **C++ core**: the `_core` extension module provides a fast Eigen-based optimiser,
  a bar-by-bar backtest engine, and a rolling z-score calculator.  The Python
  strategies include pure-Python fallbacks so the C++ build is optional.

---

## License

MIT
