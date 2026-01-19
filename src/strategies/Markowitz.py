import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

def optimize_portfolio(cov_matrix, expected_returns, target_returns, short_selling=False):
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values
    
    P = matrix(cov_matrix)
    q = matrix(np.zeros(expected_returns.shape[0]))
    A_list = [np.ones(expected_returns.shape[0])]
    b_list = [1.0]
    A_list.append(expected_returns.flatten())
    b_list.append(target_returns)
    A = matrix(np.vstack(A_list))
    b = matrix(b_list)
    G = None
    h = None
    if not short_selling:
        G = matrix(-np.eye(expected_returns.shape[0]))
        h = matrix(np.zeros(expected_returns.shape[0]))
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    weights = np.array(solution['x']).flatten()
    return weights

def calculate_efficient_frontier(cov_matrix, expected_returns, num_portfolios=100, short_selling=False):
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_portfolios)
    weights_list = []
    std_list = []
    for target in target_returns:
        weights = optimize_portfolio(cov_matrix, expected_returns, target, short_selling)
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


def main():
    ...