import numpy as np
import pandas as pd
from typing import List, Optional, Union
from data.fetch_data import fetch_stock_data, fetch_stock_info, fetch_risk_free_rate
from strategies.Markowitz import optimize_portfolio, calculate_efficient_frontier, portfolio_performance, sharpe_ratio, best_sharpe_ratio
from optimization import read_json

data = fetch_stock_data(["AAPL", "MSFT", "NVDA", "GLD", "GOOGL"], period="5y")
prices = data['Close']
returns = prices.pct_change().dropna()
expected_returns = returns.mean()
cov_matrix = returns.cov()

risk_free_rate = fetch_risk_free_rate() / 100 / 252

print(best_sharpe_ratio(calculate_efficient_frontier(cov_matrix, expected_returns), risk_free_rate))

