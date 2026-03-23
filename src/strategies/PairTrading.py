"""Cointegration-based pair trading strategy.

Steps:
1. Test all pairs for cointegration using ``statsmodels.tsa.stattools.coint``.
2. Select pairs below p-value threshold (default 0.05).
3. Compute the spread as OLS residual of the pair regression.
4. Compute rolling z-score of the spread.
5. Generate signals:
   - Enter long/short when |z| > 2.
   - Exit when |z| < 0.5.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

from ..data.data_engineering import simple_returns
from ..results.trading_book import TradingBook
from .base_trading import TradingStrategy


# ---------------------------------------------------------------------------
# Pair-trading helpers
# ---------------------------------------------------------------------------


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    p_threshold: float = 0.05,
) -> list[tuple[str, str, float]]:
    """Test all ticker pairs for cointegration.

    Parameters
    ----------
    prices:
        Price DataFrame (index=date, columns=tickers).
    p_threshold:
        Maximum p-value for a pair to be considered cointegrated (default 0.05).

    Returns
    -------
    list[tuple[str, str, float]]
        List of ``(ticker_a, ticker_b, p_value)`` for cointegrated pairs.
    """
    tickers = list(prices.columns)
    pairs: list[tuple[str, str, float]] = []

    for t1, t2 in itertools.combinations(tickers, 2):
        s1 = prices[t1].dropna()
        s2 = prices[t2].dropna()
        common_idx = s1.index.intersection(s2.index)
        if len(common_idx) < 50:
            continue
        try:
            _, pvalue, _ = coint(s1.loc[common_idx], s2.loc[common_idx])
            if pvalue < p_threshold:
                pairs.append((t1, t2, float(pvalue)))
        except Exception:
            continue

    pairs.sort(key=lambda x: x[2])
    return pairs


def compute_spread(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> pd.Series:
    """Compute the OLS spread between two price series.

    The spread is the residual of regressing *ticker_a* on *ticker_b*.

    Parameters
    ----------
    prices:
        Price DataFrame.
    ticker_a:
        Dependent variable ticker.
    ticker_b:
        Independent variable ticker.

    Returns
    -------
    pd.Series
        Spread series (residuals from OLS regression).
    """
    y = prices[ticker_a].dropna()
    x = prices[ticker_b].dropna()
    common = y.index.intersection(x.index)
    y, x = y.loc[common], x.loc[common]

    x_const = add_constant(x)
    model = OLS(y, x_const).fit()
    spread = y - model.predict(x_const)
    spread.name = f"spread_{ticker_a}_{ticker_b}"
    return spread


def rolling_zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    """Compute the rolling z-score of a spread series.

    Parameters
    ----------
    spread:
        Spread time series.
    window:
        Rolling window in days (default 30).

    Returns
    -------
    pd.Series
        Z-score series.
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.Series:
    """Generate long/short/flat signals from a z-score series.

    Signals:
    * +1 when z < -entry_threshold  (spread is cheap → long spread)
    * -1 when z > +entry_threshold  (spread is rich → short spread)
    *  0 when |z| < exit_threshold  (exit / flat)

    Parameters
    ----------
    zscore:
        Z-score series.
    entry_threshold:
        Z-score magnitude to trigger entry (default 2.0).
    exit_threshold:
        Z-score magnitude to trigger exit (default 0.5).

    Returns
    -------
    pd.Series
        Signal series (+1, -1, or 0).
    """
    signals = pd.Series(np.nan, index=zscore.index)
    position = 0

    for date, z in zscore.items():
        if np.isnan(z):
            signals[date] = 0
            continue

        if position == 0:
            if z < -entry_threshold:
                position = 1
            elif z > entry_threshold:
                position = -1
        elif position == 1:
            if abs(z) < exit_threshold:
                position = 0
        elif position == -1:
            if abs(z) < exit_threshold:
                position = 0

        signals[date] = position

    return signals.fillna(0)


# ---------------------------------------------------------------------------
# PairTrading Strategy class
# ---------------------------------------------------------------------------


class PairTradingStrategy(TradingStrategy):
    """Cointegration-based pair trading.

    Parameters
    ----------
    name:
        Strategy name.
    tickers:
        Asset tickers to screen for cointegrated pairs.
    start:
        ISO-8601 start date.
    end:
        ISO-8601 end date.
    prices:
        Asset price DataFrame (index=date, columns=tickers).
    p_threshold:
        Maximum cointegration p-value (default 0.05).
    zscore_window:
        Rolling z-score window in days (default 30).
    entry_threshold:
        Z-score magnitude to enter a position (default 2.0).
    exit_threshold:
        Z-score magnitude to exit a position (default 0.5).
    """

    def __init__(
        self,
        name: str,
        tickers: list[str],
        start: str,
        end: str,
        prices: pd.DataFrame,
        p_threshold: float = 0.05,
        zscore_window: int = 30,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
    ) -> None:
        """Initialise with price data."""
        super().__init__(name, tickers, start, end)
        self.prices = prices
        self.p_threshold = p_threshold
        self.zscore_window = zscore_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def run(self) -> TradingBook:
        """Find cointegrated pairs and generate current signals.

        Returns
        -------
        TradingBook
            TradingBook with ``positions`` containing the most recent signal
            for each pair.
        """
        pairs = find_cointegrated_pairs(self.prices, self.p_threshold)
        positions: dict[str, float] = {t: 0.0 for t in self.tickers}

        for t1, t2, _ in pairs:
            spread = compute_spread(self.prices, t1, t2)
            zscore = rolling_zscore(spread, self.zscore_window)
            signals = generate_signals(
                zscore, self.entry_threshold, self.exit_threshold
            )
            latest_signal = float(signals.iloc[-1]) if not signals.empty else 0.0

            # Long the spread: long t1, short t2
            positions[t1] = positions.get(t1, 0.0) + latest_signal
            positions[t2] = positions.get(t2, 0.0) - latest_signal

        book = TradingBook(
            name=self.name,
            tickers=self.tickers,
            positions=positions,
        )
        self._last_result = book
        return book

    def backtest(self) -> TradingBook:
        """Full historical backtest with positions_history and pnl_history.

        Returns
        -------
        TradingBook
            TradingBook with complete ``positions_history`` and ``pnl_history``.
        """
        if "date" in self.prices.columns:
            prices_idx = self.prices.set_index("date")
        else:
            prices_idx = self.prices.copy()
        prices_idx.index = pd.to_datetime(prices_idx.index)
        prices_idx = prices_idx.sort_index()

        pairs = find_cointegrated_pairs(prices_idx, self.p_threshold)
        if not pairs:
            empty_book = TradingBook(name=self.name, tickers=self.tickers)
            self._last_result = empty_book
            return empty_book

        # Build positions_history and pnl for each pair
        all_positions = pd.DataFrame(0.0, index=prices_idx.index, columns=self.tickers)
        daily_pnl = pd.Series(0.0, index=prices_idx.index)

        daily_returns = simple_returns(prices_idx)

        for t1, t2, _ in pairs:
            if t1 not in prices_idx.columns or t2 not in prices_idx.columns:
                continue

            spread = compute_spread(prices_idx, t1, t2)
            zscore = rolling_zscore(spread, self.zscore_window)
            signals = generate_signals(zscore, self.entry_threshold, self.exit_threshold)

            # Accumulate positions and P&L
            aligned = signals.reindex(prices_idx.index).fillna(0)
            all_positions[t1] += aligned
            all_positions[t2] -= aligned

            # P&L: signal × (return_t1 - return_t2) per day
            pair_rets = (
                daily_returns[t1].fillna(0) - daily_returns[t2].fillna(0)
            )
            daily_pnl += aligned.shift(1).fillna(0) * pair_rets

        last_positions = {
            t: float(all_positions[t].iloc[-1]) for t in self.tickers if t in all_positions
        }

        book = TradingBook(
            name=self.name,
            tickers=self.tickers,
            positions=last_positions,
            positions_history=all_positions,
            pnl_history=daily_pnl,
        )
        self._last_result = book
        return book
