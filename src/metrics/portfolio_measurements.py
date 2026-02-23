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
    
    portfolio_values = pd.Series(index=prices_copy.index, dtype=float)
    portfolio_returns = pd.Series(index=prices_copy.index, dtype=float)
    
    rebalance_dates = weights_clean.index
    current_value = initial_value
    
    for i in range(len(rebalance_dates)):
        target_date = rebalance_dates[i]
        next_target_date = rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else prices_copy.index[-1]
        
        available_dates = prices_copy.index[prices_copy.index >= target_date]
        if len(available_dates) == 0:
            continue
        start_date = available_dates[0]
        
        available_end_dates = prices_copy.index[prices_copy.index >= next_target_date]
        if len(available_end_dates) == 0:
            end_date = prices_copy.index[-1]
        else:
            end_date = available_end_dates[0]
        
        weights = weights_clean.loc[target_date, asset_columns].values.astype(float)
        
        valid_mask = ~np.isnan(weights)
        weights = weights[valid_mask]
        valid_assets = np.array(asset_columns)[valid_mask]
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            continue
        
        rebalance_prices = prices_copy.loc[start_date, valid_assets]
        shares = {}
        for j, asset in enumerate(valid_assets):
            price = rebalance_prices[asset]
            if not np.isnan(price) and price > 0:
                shares[asset] = (current_value * weights[j]) / price
            else:
                shares[asset] = 0

        period_prices = prices_copy.loc[start_date:end_date, valid_assets]
        prev_value = None
        
        for date in period_prices.index:
            daily_prices = period_prices.loc[date]
            portfolio_value = sum(shares.get(asset, 0) * daily_prices[asset] 
                                 for asset in valid_assets 
                                 if not np.isnan(daily_prices[asset]))
            
            portfolio_values[date] = portfolio_value

            if prev_value is not None and prev_value > 0:
                portfolio_returns[date] = (portfolio_value - prev_value) / prev_value
            else:
                portfolio_returns[date] = 0.0
            
            prev_value = portfolio_value

        current_value = portfolio_value
    
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


def compound_growth_rate(prices: pd.DataFrame, 
                         duration: int
                        ) -> float:
    """
    Compute The CGR: compound growth rate

    :param prices: Assets' historical prices
    :type prices: pd.DataFrame
    :param duration: The duration of the prices
    :type duration: int

    :returns cgr: The compound growth rate
    :rtype cgr: float
    """
    n = len(prices)
    cgr = (prices.iloc[-1] / prices.iloc[0]) ** (duration / n) - 1
    return cgr

