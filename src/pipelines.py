import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .strategies.Markowitz import l2_optimization
from .metrics.portfolio_measurements import realized_returns
from .data.stock_prices import get_stock_prices
from .data.data_engineering import format_portfolio, sector_diversification
from .visualization import portfolio_analysis, market_comparison

def portfolio_report(weights: pd.DataFrame, 
                     prices: pd.DataFrame, 
                     portfolio_value: float = 100
                    ) -> None:
    """
    Create a full and detailed report on the portfolio created: backtesting, graphs, diversification, weights reparitions

    :param weights: The weights of the portfolio over time
    :type weights: pd.DataFrame
    :param prices: The assets' historical prices
    :type prices: pd.DataFrame
    :param portfolio_value: [Optional] The monetary value invested in the portfolio, by default 100
    :type portfolio_value: float
    """
    rr = realized_returns(weights, prices)

    actual_tickers = [column for column in prices.columns if column != 'date']

    portfolio_analysis(weights, "Test L2 Optimized", prices, realized_returns=rr)

    metrics = ['sharpe_ratio', 'expected_return', 'std']
    pure_weights = weights[[col for col in weights.columns if col not in metrics]]
    portfolio_metrics = weights[metrics]

    last_weights = pure_weights.iloc[-1]
    
    last_weights = last_weights[last_weights.abs() > 1e-6]
    actual_tickers = list(last_weights.index)

    repartition = format_portfolio(last_weights, actual_tickers, portfolio_value=50000, to_txt=True, all_assets=True)

    diversification = sector_diversification(actual_tickers, True)

    plt.show()