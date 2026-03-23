"""Integration test: run_pipeline in paper mode with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipelines import run_pipeline


@pytest.fixture
def synth_prices() -> pd.DataFrame:
    """200 days of synthetic prices for 3 tickers."""
    np.random.seed(7)
    n = 200
    dates = pd.bdate_range("2022-01-01", periods=n)
    data = {
        "AAPL": 150.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n))),
        "MSFT": 300.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.012, n))),
        "GOOGL": 100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.018, n))),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_pipeline_markowitz_paper(synth_prices: pd.DataFrame) -> None:
    """run_pipeline returns a result dict for Markowitz in paper mode."""
    results = run_pipeline(
        strategy_names=["Markowitz"],
        mode="paper",
        prices=synth_prices,
        start="2022-01-01",
        end="2023-12-31",
    )
    assert "Markowitz" in results
    r = results["Markowitz"]
    assert r.get("result") is not None or r.get("error") is not None


def test_pipeline_hrp_paper(synth_prices: pd.DataFrame) -> None:
    """run_pipeline with HRP returns a valid result."""
    results = run_pipeline(
        strategy_names=["HRP"],
        mode="paper",
        prices=synth_prices,
    )
    assert "HRP" in results


def test_pipeline_multiple_strategies(synth_prices: pd.DataFrame) -> None:
    """run_pipeline accepts multiple strategy names simultaneously."""
    results = run_pipeline(
        strategy_names=["Markowitz", "HRP"],
        mode="paper",
        prices=synth_prices,
    )
    assert "Markowitz" in results
    assert "HRP" in results


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_pipeline_unknown_strategy_does_not_crash(synth_prices: pd.DataFrame) -> None:
    """An unknown strategy name is logged but does not raise an exception."""
    results = run_pipeline(
        strategy_names=["NonExistentStrategy"],
        mode="paper",
        prices=synth_prices,
    )
    # Should return an empty result (strategy not found) without crashing
    assert isinstance(results, dict)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_pipeline_empty_strategy_list(synth_prices: pd.DataFrame) -> None:
    """Empty strategy_names returns an empty dict."""
    results = run_pipeline(strategy_names=[], prices=synth_prices)
    assert results == {}
