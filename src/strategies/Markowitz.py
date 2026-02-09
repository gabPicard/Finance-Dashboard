import numpy as np
import pandas as pd
from qpsolvers import solve_qp

def optimize_portfolio(cov_matrix, expected_returns, target_returns, short_selling=False, max_weight=0.8):
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values

    P = cov_matrix
    q = -expected_returns
    
    A = np.ones((1, len(expected_returns)))
    b = np.array([1.0])
    
    G_list = []
    h_list = []
    
    if not short_selling:
        G_list.append(-np.eye(len(expected_returns)))
        h_list.append(np.zeros(len(expected_returns)))
    
    if max_weight is not None:
        G_list.append(np.eye(len(expected_returns)))
        h_list.append(np.full(len(expected_returns), max_weight))
    
    if G_list:
        G = np.vstack(G_list)
        h = np.concatenate(h_list)
    else:
        G = None
        h = None
  
    weights = solve_qp(P, q, G, h, A, b, solver="ecos")
    
    return weights

def calculate_efficient_frontier(cov_matrix, expected_returns, num_portfolios=100, short_selling=False, max_weight=0.8):
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_portfolios)
    weights_list = []
    std_list = []
    for target in target_returns:
        weights = optimize_portfolio(cov_matrix, expected_returns, target, short_selling, max_weight)
        weights_list.append(weights)
        std = portfolio_performance(weights, expected_returns, cov_matrix)['std']
        std_list.append(std)
    return {"weights":np.array(weights_list), "returns":target_returns, "std":std_list}

def portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    return {"return":portfolio_return, "std":portfolio_std}

def sharpe_ratio(portfolio_return, std, risk_free_rate):
    return (portfolio_return - risk_free_rate) / std

def best_sharpe_ratio(efficient_frontier, risk_free_rate):
    returns = efficient_frontier['returns']
    std_list = efficient_frontier['std']
    max_sharpe = 0
    index = 0
    for i in range (0, returns.shape[0]):
        sharpe = sharpe_ratio(returns[i], std_list[i], risk_free_rate)
        if sharpe > max_sharpe:
            max_sharpe = sharpe
            index = i
    return {"sharpe ratio": max_sharpe, 
            "weights": efficient_frontier['weights'][index], 
            "expected return": returns[index], 
            "standard deviation": std_list[index]
            }

def rolling_window(prices, risk_free_rate, rebalance_frequency, short_selling=False, max_weight=0.8):
    prices_copy = prices.copy()
    prices_copy['date'] = pd.to_datetime(prices_copy['date'])
    prices_copy = prices_copy.set_index('date').sort_index()
    
    index_rebalancement = prices_copy.resample(rebalance_frequency).last().index
    
    asset_columns = list(prices_copy.columns)
    metric_columns = ['sharpe_ratio', 'expected_return', 'std']
    all_columns = asset_columns + metric_columns
    
    weights_backtest = pd.DataFrame(index=index_rebalancement, columns=all_columns)
    
    for ind in index_rebalancement:
        price_tmp = prices_copy[:ind].tail(252)
        return_tmp = price_tmp.pct_change().dropna()
        cov_tmp = return_tmp.cov()
        exp_returns = return_tmp.mean()

        efficient_frontier = calculate_efficient_frontier(cov_tmp, exp_returns, short_selling=short_selling, max_weight=max_weight)

        best_portfolio = best_sharpe_ratio(efficient_frontier, risk_free_rate)

        weights_backtest.loc[ind, asset_columns] = best_portfolio['weights']
        
        weights_backtest.loc[ind, 'sharpe_ratio'] = best_portfolio['sharpe ratio']
        weights_backtest.loc[ind, 'expected_return'] = best_portfolio['expected return']
        weights_backtest.loc[ind, 'std'] = best_portfolio['standard deviation']
    
    return weights_backtest

def main():
    ...