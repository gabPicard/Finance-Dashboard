import numpy as np
import pandas as pd
import warnings
from qpsolvers import solve_qp

def optimize_portfolio(cov_matrix, expected_returns, target_returns=None, max_weight=0.15):
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values

    max_weight = max(0.15, len(expected_returns))

    P = cov_matrix
    q = np.zeros(len(expected_returns))
    
    A_list = [np.ones((1, len(expected_returns)))]
    b_list = [1.0]
    
    if target_returns is not None:
        A_list.append(expected_returns.reshape(1, -1))
        b_list.append(target_returns)
    
    A = np.vstack(A_list)
    b = np.array(b_list)
    
    G_list = []
    h_list = []
    
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
  
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        weights = solve_qp(P, q, G, h, A, b, solver="ecos")
    
    return weights

def calculate_efficient_frontier(cov_matrix, expected_returns, num_portfolios=100, max_weight=0.15):
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_portfolios)
    weights_list = []
    std_list = []
    valid_returns = []
    for target in target_returns:
        weights = optimize_portfolio(cov_matrix, expected_returns, target, max_weight)
        if weights is not None:
            weights_list.append(weights)
            std = portfolio_performance(weights, expected_returns, cov_matrix)['std']
            std_list.append(std)
            valid_returns.append(target)
    return {"weights":np.array(weights_list), "returns":np.array(valid_returns), "std":std_list}

def portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    return {"return":portfolio_return, "std":portfolio_std}

def sharpe_ratio(portfolio_return, std, risk_free_rate):
    if std == 0 or np.isnan(std):
        return 0.0
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

def rolling_window(prices, risk_free_rate, rebalance_frequency, strategy="Best sharpe", max_weight=0.15):
    prices_copy = prices.copy()
    prices_copy['date'] = pd.to_datetime(prices_copy['date'])
    prices_copy = prices_copy.set_index('date').sort_index()
    
    index_rebalancement = prices_copy.resample(rebalance_frequency).last().index
    
    asset_columns = list(prices_copy.columns)
    metric_columns = ['sharpe_ratio', 'expected_return', 'std']
    all_columns = asset_columns + metric_columns

    max_weight = max(0.15, 1/len(asset_columns))
    
    weights_backtest = pd.DataFrame(index=index_rebalancement, columns=all_columns)
    last_valid_weights = None  # Track last successful optimization
    
    for ind in index_rebalancement:
        price_tmp = prices_copy[:ind].tail(252)
        return_tmp = price_tmp.pct_change(fill_method=None).dropna()
        
        try:
            
            cov_tmp = return_tmp.cov() * 252
            exp_returns = return_tmp.mean() * 252
            
            eigenvalues = np.linalg.eigvals(cov_tmp)
            if np.any(eigenvalues <= 0):
                warnings.warn(f"Skipping {ind}: Covariance matrix is not positive definite - using previous weights")
                if last_valid_weights is not None:
                    weights_backtest.loc[ind, asset_columns] = last_valid_weights['weights']
                    weights_backtest.loc[ind, 'sharpe_ratio'] = last_valid_weights['sharpe_ratio']
                    weights_backtest.loc[ind, 'expected_return'] = last_valid_weights['expected_return']
                    weights_backtest.loc[ind, 'std'] = last_valid_weights['std']
                continue
                
        except Exception as e:
            warnings.warn(f"Skipping {ind}: Data validation failed - {str(e)} (Available data rows: {len(return_tmp)}, Assets: {len(return_tmp.columns)}) - using previous weights")
            if last_valid_weights is not None:
                weights_backtest.loc[ind, asset_columns] = last_valid_weights['weights']
                weights_backtest.loc[ind, 'sharpe_ratio'] = last_valid_weights['sharpe_ratio']
                weights_backtest.loc[ind, 'expected_return'] = last_valid_weights['expected_return']
                weights_backtest.loc[ind, 'std'] = last_valid_weights['std']
            continue

        if strategy == "Best sharpe":
            efficient_frontier = calculate_efficient_frontier(cov_tmp, exp_returns, max_weight=max_weight)
            
            if len(efficient_frontier['weights']) > 0:
                best_portfolio = best_sharpe_ratio(efficient_frontier, risk_free_rate)

                weights_backtest.loc[ind, asset_columns] = 0.0
                for i, asset in enumerate(return_tmp.columns):
                    weights_backtest.loc[ind, asset] = best_portfolio['weights'][i]
        
                weights_backtest.loc[ind, 'sharpe_ratio'] = best_portfolio['sharpe ratio']
                weights_backtest.loc[ind, 'expected_return'] = best_portfolio['expected return']
                weights_backtest.loc[ind, 'std'] = best_portfolio['standard deviation']
                
                # Store as last valid weights
                last_valid_weights = {
                    'weights': best_portfolio['weights'],
                    'sharpe_ratio': best_portfolio['sharpe ratio'],
                    'expected_return': best_portfolio['expected return'],
                    'std': best_portfolio['standard deviation']
                }
            
            else:
                warnings.warn(f"No efficient frontier found - using previous weights")
                if last_valid_weights is not None:
                    weights_backtest.loc[ind, asset_columns] = last_valid_weights['weights']
                    weights_backtest.loc[ind, 'sharpe_ratio'] = last_valid_weights['sharpe_ratio']
                    weights_backtest.loc[ind, 'expected_return'] = last_valid_weights['expected_return']
                    weights_backtest.loc[ind, 'std'] = last_valid_weights['std']
        
        elif strategy == "Lowest std":
            opt_weights = optimize_portfolio(cov_tmp, exp_returns, target_return=None, max_weight=max_weight)
            
            if opt_weights is not None:
                performance = portfolio_performance(opt_weights, exp_returns, cov_tmp)
                sharpe = sharpe_ratio(performance['return'], performance['std'], risk_free_rate)

                weights_backtest.loc[ind, asset_columns] = 0.0
                for i, asset in enumerate(return_tmp.columns):
                    weights_backtest.loc[ind, asset] = opt_weights[i]
            
                weights_backtest.loc[ind, 'sharpe_ratio'] = sharpe
                weights_backtest.loc[ind, 'expected_return'] = performance['return']
                weights_backtest.loc[ind, 'std'] = performance['std']
                
                # Store as last valid weights
                last_valid_weights = {
                    'weights': opt_weights,
                    'sharpe_ratio': sharpe,
                    'expected_return': performance['return'],
                    'std': performance['std']
                }

    return weights_backtest

def l2_optimization(prices, risk_free_rate, rebalance_frequency, rho, gamma, max_weight=0.15):
    prices_copy = prices.copy()
    prices_copy['date'] = pd.to_datetime(prices_copy['date'])
    prices_copy = prices_copy.set_index('date').sort_index()
    
    index_rebalancement = prices_copy.resample(rebalance_frequency).last().index
    
    asset_columns = list(prices_copy.columns)
    metric_columns = ['sharpe_ratio', 'expected_return', 'std']
    all_columns = asset_columns + metric_columns

    max_weight = max(0.15, 1/len(asset_columns))
    
    weights_backtest = pd.DataFrame(index=index_rebalancement, columns=all_columns)
    weights_old = np.array([1/len(asset_columns) for _ in range(len(asset_columns))])
    
    for ind in index_rebalancement:
        price_tmp = prices_copy[:ind].tail(252)
        return_tmp = price_tmp.pct_change(fill_method=None).dropna()
        
        try:
            
            cov_matrix = return_tmp.cov() * 252
            expected_returns = return_tmp.mean() * 252
            
            eigenvalues = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvalues <= 0):
                warnings.warn(f"Skipping {ind}: Covariance matrix is not positive definite - using previous weights")
                weights_backtest.loc[ind, asset_columns] = weights_old
                weights_backtest.loc[ind, 'sharpe_ratio'] = np.nan
                weights_backtest.loc[ind, 'expected_return'] = np.nan
                weights_backtest.loc[ind, 'std'] = np.nan
                continue
                
        except Exception as e:
            warnings.warn(f"Skipping {ind}: Data validation failed - {str(e)} (Available data rows: {len(return_tmp)}, Assets: {len(return_tmp.columns)}) - using previous weights")
            weights_backtest.loc[ind, asset_columns] = weights_old
            weights_backtest.loc[ind, 'sharpe_ratio'] = np.nan
            weights_backtest.loc[ind, 'expected_return'] = np.nan
            weights_backtest.loc[ind, 'std'] = np.nan
            continue

        P = cov_matrix
        q = np.zeros(len(expected_returns))

        A_list = [np.ones((1, len(expected_returns)))]
        b_list = [1.0]

        A = np.vstack(A_list)
        b = np.array(b_list)

        G_list = []
        h_list = []

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

        In = np.eye((len(expected_returns)))

        P += 2 * rho * In
        q = -gamma * expected_returns + 2 * rho * weights_old.T

    
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            opt_weights = solve_qp(P, q, G, h, A, b, solver="ecos")
        
        performance = portfolio_performance(opt_weights, expected_returns, cov_matrix)
        sharpe = sharpe_ratio(performance['return'], performance['std'], risk_free_rate)
        weights_backtest.loc[ind, asset_columns] = 0.0
        for i, asset in enumerate(return_tmp.columns):
            weights_backtest.loc[ind, asset] = opt_weights[i]
    
        weights_backtest.loc[ind, 'sharpe_ratio'] = sharpe
        weights_backtest.loc[ind, 'expected_return'] = performance['return']
        weights_backtest.loc[ind, 'std'] = performance['std']

        weights_old = opt_weights
    
    return weights_backtest

def find_best_params(prices, risk_free_rate, deciding_value="Lowest std"):
    rebalance_frequency = 'QE'
    max_weight = 0.15

    gammas = np.linspace(0.01, 0.05, 10)
    rhos = np.linspace(0.01, 1, 100)

    metrics = ['Return', 'Sharpe Ratio', 'Standard Deviation']
    params = pd.DataFrame(
        index=pd.Index(gammas, name="gamma"),
        columns=pd.MultiIndex.from_product([rhos, metrics], names=["rho", "metric"]),
    )

    for ind_gammas in gammas:
        for ind_rhos in rhos:
            l2_results = l2_optimization(prices, risk_free_rate, rebalance_frequency, ind_rhos, ind_gammas, max_weight=max_weight)

            if l2_results is None or l2_results.empty:
                continue

            last_metrics = l2_results[['expected_return', 'sharpe_ratio', 'std']].dropna()
            if last_metrics.empty:
                continue

            last_metrics = last_metrics.iloc[-1]

            params.loc[ind_gammas, (ind_rhos, 'Return')] = last_metrics['expected_return']
            params.loc[ind_gammas, (ind_rhos, 'Sharpe Ratio')] = last_metrics['sharpe_ratio']
            params.loc[ind_gammas, (ind_rhos, 'Standard Deviation')] = last_metrics['std']
    
    return params

def main():
    ...