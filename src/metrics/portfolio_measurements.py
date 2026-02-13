import numpy as np
import pandas as pd

def realized_returns(weights_backtest, prices, initial_value=100):
    """
    Calculate realized portfolio returns based on backtested weights.
    
    :params weights_backtest: DataFrame with weights at each rebalancing date (from rolling_window)
    :type weights_backtest: pd.Dataframe
    :params prices: Price data with date index
    :type prices: pd.Dataframe
        
    returns: results: dict
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
    
    asset_columns = [col for col in weights_backtest.columns 
                     if col not in ['sharpe_ratio', 'expected_return', 'std']]
    
    weights_clean = weights_backtest.dropna(how='all', subset=asset_columns)
    
    if len(weights_clean) == 0:
        raise ValueError("No valid rebalancing periods found in weights_backtest")
    
    daily_returns = prices_copy.pct_change(fill_method=None)
    
    portfolio_values = pd.Series(index=daily_returns.index, dtype=float)
    portfolio_returns = pd.Series(index=daily_returns.index, dtype=float)
    
    rebalance_dates = weights_clean.index
    current_value = initial_value
    
    for i in range(len(rebalance_dates)):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else daily_returns.index[-1]
        
        weights = weights_clean.loc[start_date, asset_columns].values.astype(float)
        
        valid_mask = ~np.isnan(weights)
        weights = weights[valid_mask]
        valid_assets = np.array(asset_columns)[valid_mask]
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            continue
        
        period_returns = daily_returns.loc[start_date:end_date, valid_assets]
        
        for date in period_returns.index:
            if date == start_date:
                portfolio_values[date] = current_value
                portfolio_returns[date] = 0.0
            else:
                daily_ret = period_returns.loc[date].values
                if np.any(np.isnan(daily_ret)):
                    daily_ret = np.nan_to_num(daily_ret, nan=0.0)
                
                portfolio_return = np.dot(weights, daily_ret)
                portfolio_returns[date] = portfolio_return
                current_value = current_value * (1 + portfolio_return)
                portfolio_values[date] = current_value
    
    portfolio_values = portfolio_values.dropna()
    portfolio_returns = portfolio_returns.dropna()
    
    cumulative_return = (portfolio_values.iloc[-1] - initial_value) / initial_value
    
    n_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    annualized_return = (portfolio_values.iloc[-1] / initial_value) ** (1 / n_years) - 1 if n_years > 0 else 0
    
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