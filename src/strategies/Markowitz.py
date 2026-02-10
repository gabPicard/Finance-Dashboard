import numpy as np
import pandas as pd
import warnings
from qpsolvers import solve_qp

def validate_and_clean_data(returns, min_observations=100, min_variance=1e-8):
    """
    Validate and clean return data before optimization.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    min_observations : int
        Minimum number of non-NaN observations required per asset
    min_variance : float
        Minimum variance required per asset
        
    Returns:
    --------
    cleaned_returns : pd.DataFrame
        Cleaned returns with problematic assets removed
    excluded_assets : list
        List of excluded asset names with reasons
    """
    excluded_assets = []
    valid_columns = []
    
    for col in returns.columns:
        series = returns[col]
        
        # Check for NaN values
        nan_count = series.isna().sum()
        if nan_count > len(series) * 0.2:  # More than 20% NaN
            excluded_assets.append((col, f"Too many NaN values ({nan_count}/{len(series)})"))
            continue
            
        # Check for sufficient observations
        valid_obs = series.dropna()
        if len(valid_obs) < min_observations:
            excluded_assets.append((col, f"Insufficient observations ({len(valid_obs)} < {min_observations})"))
            continue
            
        # Check for zero or near-zero variance
        variance = valid_obs.var()
        if variance < min_variance or np.isnan(variance):
            excluded_assets.append((col, f"Zero or near-zero variance ({variance})"))
            continue
            
        # Check for constant values
        if valid_obs.nunique() <= 1:
            excluded_assets.append((col, "Constant values"))
            continue
            
        valid_columns.append(col)
    
    if len(valid_columns) == 0:
        raise ValueError("No valid assets remaining after data validation")
    
    cleaned_returns = returns[valid_columns].copy()
    
    # Forward fill then backward fill remaining NaN values
    cleaned_returns = cleaned_returns.ffill().bfill()
    
    # Final check: drop any remaining rows with NaN
    cleaned_returns = cleaned_returns.dropna()
    
    if len(excluded_assets) > 0:
        warnings.warn(f"Excluded {len(excluded_assets)} assets due to data quality issues")
    
    return cleaned_returns, excluded_assets

def optimize_portfolio(cov_matrix, expected_returns, target_returns=None, short_selling=False, max_weight=0.8):
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values

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
  
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        weights = solve_qp(P, q, G, h, A, b, solver="ecos")
    
    return weights

def calculate_efficient_frontier(cov_matrix, expected_returns, num_portfolios=100, short_selling=False, max_weight=0.8):
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_portfolios)
    weights_list = []
    std_list = []
    valid_returns = []
    for target in target_returns:
        weights = optimize_portfolio(cov_matrix, expected_returns, target, short_selling, max_weight)
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
        return 0.0  # Return 0 if std is zero to avoid division by zero
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

def rolling_window(prices, risk_free_rate, rebalance_frequency, strategy="Best sharpe", short_selling=False, max_weight=0.8):
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
        return_tmp = price_tmp.pct_change(fill_method=None).dropna()
        
        # Validate and clean data before optimization
        # Adjust min_observations based on available data
        available_rows = len(return_tmp)
        min_obs = max(20, int(available_rows * 0.2))  # At least 20% of available data
        
        try:
            cleaned_returns, excluded = validate_and_clean_data(return_tmp, min_observations=min_obs, min_variance=1e-10)
            
            if len(cleaned_returns.columns) < 2:
                warnings.warn(f"Skipping {ind}: Less than 2 valid assets remaining")
                continue
            
            # Annualize returns and covariance (252 trading days per year)
            cov_tmp = cleaned_returns.cov() * 252
            exp_returns = cleaned_returns.mean() * 252
            
            # Check if covariance matrix is positive definite
            eigenvalues = np.linalg.eigvals(cov_tmp)
            if np.any(eigenvalues <= 0):
                warnings.warn(f"Skipping {ind}: Covariance matrix is not positive definite")
                continue
                
        except Exception as e:
            warnings.warn(f"Skipping {ind}: Data validation failed - {str(e)} (Available data rows: {len(return_tmp)}, Assets: {len(return_tmp.columns)})")
            continue

        if strategy == "Best sharpe":
            efficient_frontier = calculate_efficient_frontier(cov_tmp, exp_returns, short_selling=short_selling, max_weight=max_weight)
            
            if len(efficient_frontier['weights']) > 0:
                best_portfolio = best_sharpe_ratio(efficient_frontier, risk_free_rate)

                # Initialize all weights to 0
                weights_backtest.loc[ind, asset_columns] = 0.0
                # Assign weights only to valid assets
                for i, asset in enumerate(cleaned_returns.columns):
                    weights_backtest.loc[ind, asset] = best_portfolio['weights'][i]
        
                weights_backtest.loc[ind, 'sharpe_ratio'] = best_portfolio['sharpe ratio']
                weights_backtest.loc[ind, 'expected_return'] = best_portfolio['expected return']
                weights_backtest.loc[ind, 'std'] = best_portfolio['standard deviation']
        
        elif strategy == "Lowest std":
            opt_weights = optimize_portfolio(cov_tmp, exp_returns, None, short_selling, max_weight)
            
            if opt_weights is not None:
                performance = portfolio_performance(opt_weights, exp_returns, cov_tmp)
                sharpe = sharpe_ratio(performance['return'], performance['std'], risk_free_rate)

                # Initialize all weights to 0
                weights_backtest.loc[ind, asset_columns] = 0.0
                # Assign weights only to valid assets
                for i, asset in enumerate(cleaned_returns.columns):
                    weights_backtest.loc[ind, asset] = opt_weights[i]
            
                weights_backtest.loc[ind, 'sharpe_ratio'] = sharpe
                weights_backtest.loc[ind, 'expected_return'] = performance['return']
                weights_backtest.loc[ind, 'std'] = performance['std']


    
    return weights_backtest

def calculate_realized_returns(weights_backtest, prices, initial_value=100):
    """
    Calculate realized portfolio returns based on backtested weights.
    
    Parameters:
    -----------
    weights_backtest : pd.DataFrame
        DataFrame with weights at each rebalancing date (from rolling_window)
    prices : pd.DataFrame
        Price data with date index
    initial_value : float
        Initial portfolio value (default 100)
        
    Returns:
    --------
    results : dict
        - 'portfolio_value': Series of portfolio values over time
        - 'portfolio_returns': Series of portfolio returns over time
        - 'cumulative_return': Total return from start to end
        - 'annualized_return': Annualized return
        - 'avg_sharpe': Average Sharpe ratio
        - 'avg_std': Average standard deviation
    """
    prices_copy = prices.copy()
    if 'date' in prices_copy.columns:
        prices_copy['date'] = pd.to_datetime(prices_copy['date'])
        prices_copy = prices_copy.set_index('date').sort_index()
    
    # Get asset columns (exclude metrics)
    asset_columns = [col for col in weights_backtest.columns 
                     if col not in ['sharpe_ratio', 'expected_return', 'std']]
    
    # Remove rows with all NaN weights
    weights_clean = weights_backtest.dropna(how='all', subset=asset_columns)
    
    if len(weights_clean) == 0:
        raise ValueError("No valid rebalancing periods found in weights_backtest")
    
    # Calculate daily returns
    daily_returns = prices_copy.pct_change(fill_method=None)
    
    # Initialize portfolio value series
    portfolio_values = pd.Series(index=daily_returns.index, dtype=float)
    portfolio_returns = pd.Series(index=daily_returns.index, dtype=float)
    
    rebalance_dates = weights_clean.index
    current_value = initial_value
    
    for i in range(len(rebalance_dates)):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else daily_returns.index[-1]
        
        # Get weights for this period
        weights = weights_clean.loc[start_date, asset_columns].values.astype(float)
        
        # Filter out NaN weights and corresponding assets
        valid_mask = ~np.isnan(weights)
        weights = weights[valid_mask]
        valid_assets = np.array(asset_columns)[valid_mask]
        
        # Normalize weights (in case they don't sum to 1 due to filtering)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            continue
        
        # Get returns for this period
        period_returns = daily_returns.loc[start_date:end_date, valid_assets]
        
        # Calculate portfolio returns for each day
        for date in period_returns.index:
            if date == start_date:
                portfolio_values[date] = current_value
                portfolio_returns[date] = 0.0
            else:
                daily_ret = period_returns.loc[date].values
                # Handle NaN in daily returns
                if np.any(np.isnan(daily_ret)):
                    daily_ret = np.nan_to_num(daily_ret, nan=0.0)
                
                portfolio_return = np.dot(weights, daily_ret)
                portfolio_returns[date] = portfolio_return
                current_value = current_value * (1 + portfolio_return)
                portfolio_values[date] = current_value
    
    # Calculate summary statistics
    portfolio_values = portfolio_values.dropna()
    portfolio_returns = portfolio_returns.dropna()
    
    cumulative_return = (portfolio_values.iloc[-1] - initial_value) / initial_value
    
    # Annualized return
    n_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    annualized_return = (portfolio_values.iloc[-1] / initial_value) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Average metrics from backtest
    avg_sharpe = weights_clean['sharpe_ratio'].astype(float).mean()
    avg_std = weights_clean['std'].astype(float).mean()
    
    return {
        'portfolio_value': portfolio_values,
        'portfolio_returns': portfolio_returns,
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'avg_sharpe': avg_sharpe,
        'avg_std': avg_std,
        'final_value': portfolio_values.iloc[-1],
        'initial_value': initial_value
    }

def diagnose_data_quality(prices, window_size=252):
    """
    Diagnose data quality issues for all assets in the price dataframe.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with date index
    window_size : int
        Rolling window size to check
        
    Returns:
    --------
    report : pd.DataFrame
        Summary report of data quality issues per asset
    """
    prices_copy = prices.copy()
    if 'date' in prices_copy.columns:
        prices_copy['date'] = pd.to_datetime(prices_copy['date'])
        prices_copy = prices_copy.set_index('date').sort_index()
    
    returns = prices_copy.pct_change().dropna()
    
    report_data = []
    for col in returns.columns:
        series = returns[col]
        
        nan_count = series.isna().sum()
        nan_pct = (nan_count / len(series)) * 100
        valid_obs = len(series.dropna())
        variance = series.var()
        mean_return = series.mean()
        unique_values = series.nunique()
        
        issues = []
        if nan_pct > 20:
            issues.append(f"High NaN ({nan_pct:.1f}%)")
        if valid_obs < window_size * 0.5:
            issues.append(f"Few observations ({valid_obs})")
        if variance < 1e-8:
            issues.append(f"Zero variance ({variance:.2e})")
        if unique_values <= 1:
            issues.append("Constant values")
            
        report_data.append({
            'Asset': col,
            'Valid Obs': valid_obs,
            'NaN %': f"{nan_pct:.2f}",
            'Variance': f"{variance:.6e}",
            'Mean Return': f"{mean_return:.6f}",
            'Unique Values': unique_values,
            'Issues': ', '.join(issues) if issues else 'OK'
        })
    
    report_df = pd.DataFrame(report_data)
    return report_df

def main():
    ...