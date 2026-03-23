-- Finance Dashboard Database Schema
-- Matches SQLAlchemy ORM models exactly

CREATE TABLE IF NOT EXISTS markets (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL UNIQUE,
    exchange TEXT NOT NULL,
    region  TEXT NOT NULL,
    currency TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_markets_name ON markets (name);

CREATE TABLE IF NOT EXISTS corporations (
    ticker      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    sector      TEXT,
    industry    TEXT,
    market_cap  REAL,
    market_id   INTEGER REFERENCES markets(id) ON DELETE SET NULL,
    description TEXT,
    ipo_date    TEXT
);

CREATE INDEX IF NOT EXISTS ix_corporations_market_id ON corporations (market_id);

CREATE TABLE IF NOT EXISTS price_history (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker  TEXT NOT NULL REFERENCES corporations(ticker) ON DELETE CASCADE,
    date    TEXT NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL NOT NULL,
    volume  REAL,
    UNIQUE (ticker, date)
);

CREATE INDEX IF NOT EXISTS ix_price_history_ticker ON price_history (ticker);
CREATE INDEX IF NOT EXISTS ix_price_history_date   ON price_history (date);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name   TEXT NOT NULL,
    run_at          TEXT NOT NULL,
    tickers         TEXT NOT NULL,
    sharpe          REAL,
    max_drawdown    REAL,
    total_return    REAL
);

CREATE INDEX IF NOT EXISTS ix_backtest_runs_strategy_name ON backtest_runs (strategy_name);
