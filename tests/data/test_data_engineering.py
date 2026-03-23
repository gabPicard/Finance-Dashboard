"""Tests for src.data.data_engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.data_engineering import (
    annualised_volatility,
    fix_price_anomalies,
    log_returns,
    rolling_covariance,
    rolling_correlation,
    simple_returns,
    validate_and_clean_data,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_log_returns_shape(sample_prices: pd.DataFrame) -> None:
    """log_returns returns a DataFrame with one fewer row than input."""
    rets = log_returns(sample_prices)
    assert rets.shape[0] == sample_prices.shape[0] - 1
    assert set(rets.columns) == set(sample_prices.columns)


def test_simple_returns_positive_days(sample_prices: pd.DataFrame) -> None:
    """simple_returns values have the same sign as price changes."""
    rets = simple_returns(sample_prices)
    price_diff = sample_prices.diff().dropna()
    assert ((rets > 0) == (price_diff > 0)).all().all()


def test_annualised_volatility_positive(sample_prices: pd.DataFrame) -> None:
    """annualised_volatility returns positive values for all tickers."""
    vol = annualised_volatility(sample_prices)
    assert (vol > 0).all()


def test_rolling_covariance_shape(sample_prices: pd.DataFrame) -> None:
    """rolling_covariance returns a MultiIndex DataFrame."""
    cov = rolling_covariance(sample_prices, window=60)
    # MultiIndex — check the number of tickers in the inner level
    n = len(sample_prices.columns)
    assert cov.shape[1] == n


def test_rolling_correlation_diagonal_ones(sample_prices: pd.DataFrame) -> None:
    """The last cross-section of rolling_correlation has diagonal = 1."""
    corr = rolling_correlation(sample_prices, window=60)
    last = corr.xs(corr.index.get_level_values(0)[-1], level=0)
    np.testing.assert_allclose(np.diag(last.values), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fix_price_anomalies_replaces_spike(sample_prices: pd.DataFrame) -> None:
    """fix_price_anomalies replaces a single extreme spike."""
    prices_with_spike = sample_prices.copy()
    prices_with_spike.iloc[50, 0] *= 10  # introduce spike
    corrected, excluded = fix_price_anomalies(prices_with_spike, max_daily_change=0.5)
    assert corrected.iloc[50, 0] == prices_with_spike.iloc[49, 0]
    assert len(excluded) == 0


def test_fix_price_anomalies_excludes_bad_asset(sample_prices: pd.DataFrame) -> None:
    """Assets with too many anomalies are flagged for exclusion."""
    bad = sample_prices.copy()
    for i in range(0, 50, 5):
        bad.iloc[i, 0] *= 100
    _, excluded = fix_price_anomalies(bad, max_daily_change=0.5, max_anomalies=3)
    assert sample_prices.columns[0] in excluded


def test_validate_and_clean_data_removes_constant(sample_prices: pd.DataFrame) -> None:
    """validate_and_clean_data removes a constant-valued asset."""
    prices_with_const = sample_prices.copy()
    prices_with_const["CONST"] = 100.0
    rets = simple_returns(prices_with_const)
    clean, excluded = validate_and_clean_data(rets)
    assert "CONST" not in clean.columns
    assert any("CONST" in t[0] for t in excluded)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_validate_raises_when_no_valid_assets() -> None:
    """validate_and_clean_data raises ValueError when no valid columns remain."""
    const_df = pd.DataFrame({"A": [1.0] * 200, "B": [2.0] * 200})
    with pytest.raises(ValueError, match="No valid assets"):
        validate_and_clean_data(const_df)
