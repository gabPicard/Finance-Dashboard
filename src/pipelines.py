import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from .strategies.Markowitz import l2_optimization, l2_algorithm, portfolio_performance, sharpe_ratio, MonteCarlo_portfolio
from .strategies.CAPM import capm_expected_returns
from .metrics.portfolio_measurements import realized_returns
from .data.stock_prices import get_stock_prices, get_tickers_list
from .data.data_engineering import format_portfolio, sector_diversification
from .visualization import portfolio_analysis, market_comparison
from .data.fetch_data import fetch_stock_data

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

def l2_capm(markets: str | list[str],
            rho: float,
            gamma: float, 
            rebalance_frequency: str,
            max_weight: float,
            start_date: str = None,
            end_date: str = None,
            period: str = "1y",
            interval: str = "1d"
            ) -> pd.DataFrame:
    """
    Use the L2 optimization algorithm at every rebalancing date using CAPM expected returns.
    
    For multiple markets, computes CAPM expected returns for each market separately, 
    then merges them for L2 optimization.
    
    :param markets: The name of a market, or a list of markets
    :type markets: str | list[str]
    :param rho: Represents how much the past portfolio matters in the current one
    :type rho: float
    :param gamma: L2-optimization parameter
    :type gamma: float
    :param rebalance_frequency: The frequency of weights rebalancement (e.g., 'QE', 'ME')
    :type rebalance_frequency: str
    :param max_weight: The maximum weight an asset can hold in the portfolio
    :type max_weight: float
    :param start_date: [Optional] The starting point for fetching data
    :type start_date: str
    :param end_date: [Optional] The end point for fetching data
    :type end_date: str
    :param period: [Optional] The period for fetching. By default, 1 year
    :type period: str
    :param interval: [Optional] The interval to fetch data. By default, 1 day
    :type interval: str
    
    :returns weights_backtest: A DataFrame containing portfolio weights at each rebalancing date
    :rtype weights_backtest: pd.DataFrame
    """
    
    # Get prices and market data
    if isinstance(markets, str):
        # Single market case
        prices, market_prices, risk_free_rate = get_stock_prices(markets,
                                                                 start_date=start_date,
                                                                 end_date=end_date,
                                                                 period=period,
                                                                 interval=interval)
        
        # Prepare data
        prices_copy = prices.copy()
        prices_copy['date'] = pd.to_datetime(prices_copy['date'])
        prices_copy = prices_copy.set_index('date').sort_index()
        
        market_prices_copy = market_prices.copy()
        market_prices_copy['date'] = pd.to_datetime(market_prices_copy['date'])
        market_prices_copy = market_prices_copy.set_index('date').sort_index()
        
    else:
        # Multiple markets case
        all_prices = []
        all_market_prices = {}
        risk_free_rate = None
        
        for market in markets:
            market_prices_data, market_index_prices, rfr = get_stock_prices(market,
                                                                            start_date=start_date,
                                                                            end_date=end_date,
                                                                            period=period,
                                                                            interval=interval)
            # Store prices for each market
            market_prices_tmp = market_prices_data.copy()
            market_prices_tmp['date'] = pd.to_datetime(market_prices_tmp['date'])
            market_prices_tmp = market_prices_tmp.set_index('date').sort_index()
            all_prices.append(market_prices_tmp)
            
            # Store market index prices
            market_index_tmp = market_index_prices.copy()
            market_index_tmp['date'] = pd.to_datetime(market_index_tmp['date'])
            market_index_tmp = market_index_tmp.set_index('date').sort_index()
            all_market_prices[market] = market_index_tmp
            
            # Use the first market's risk-free rate
            if risk_free_rate is None:
                risk_free_rate = rfr
        
        # Merge all prices (outer join to keep all assets)
        prices_copy = pd.concat(all_prices, axis=1)
        prices_copy = prices_copy.loc[:, ~prices_copy.columns.duplicated()]
    
    # Setup for backtesting
    index_rebalancement = prices_copy.resample(rebalance_frequency).last().index
    asset_columns = list(prices_copy.columns)
    metric_columns = ['sharpe_ratio', 'expected_return', 'std']
    all_columns = asset_columns + metric_columns
    
    if max_weight is not None:
        max_weight = max(min(max_weight, 1.0), 1/len(asset_columns))
    
    weights_backtest = pd.DataFrame(index=index_rebalancement, columns=all_columns)
    
    # Loop through each rebalancing date
    for ind in index_rebalancement:
        price_tmp = prices_copy[:ind].tail(252)
        
        if len(price_tmp) < 63:
            warnings.warn(f"Skipping {ind}: Insufficient data ({len(price_tmp)} days, need 63)")
            continue
        
        return_tmp = price_tmp.pct_change(fill_method=None).dropna()
        
        try:
            # Compute CAPM expected returns
            if isinstance(markets, str):
                # Single market case
                market_tmp = market_prices_copy[:ind].tail(252)
                expected_returns = capm_expected_returns(price_tmp, market_tmp, risk_free_rate)
            else:
                # Multiple markets case: compute CAPM for each market separately
                all_expected_returns = {}
                
                for market in markets:
                    # Get tickers for this market
                    tickers_list, _, _ = get_tickers_list(market)
                    
                    # Filter prices for this market's assets
                    market_assets = [col for col in price_tmp.columns if col in tickers_list]
                    
                    if len(market_assets) == 0:
                        continue
                    
                    market_prices_tmp = price_tmp[market_assets]
                    market_index_tmp = all_market_prices[market][:ind].tail(252)
                    
                    # Compute CAPM expected returns for this market
                    market_capm = capm_expected_returns(market_prices_tmp, market_index_tmp, risk_free_rate)
                    all_expected_returns.update(market_capm)
                
                # Merge all expected returns
                expected_returns = all_expected_returns
            
            # Convert expected returns dict to Series aligned with return_tmp columns
            # Only keep assets that have expected returns
            available_assets = [col for col in return_tmp.columns if col in expected_returns]
            if len(available_assets) == 0:
                warnings.warn(f"Skipping {ind}: No assets have CAPM expected returns - using previous weights")
                # Use previous weights or equal weights
                if len(weights_backtest.dropna(how='all')) > 0:
                    last_weights = weights_backtest[asset_columns].iloc[-1]
                    weights_backtest.loc[ind, asset_columns] = last_weights
                else:
                    weights_backtest.loc[ind, asset_columns] = 1/len(asset_columns)
                weights_backtest.loc[ind, 'sharpe_ratio'] = np.nan
                weights_backtest.loc[ind, 'expected_return'] = np.nan
                weights_backtest.loc[ind, 'std'] = np.nan
                continue
            
            # Filter to only available assets
            return_tmp = return_tmp[available_assets]
            cov_matrix = return_tmp.cov() * 252
            expected_returns_series = pd.Series({asset: expected_returns[asset] for asset in available_assets})
            
            # Prepare weights_old for current available assets
            # Extract weights for available assets from previous iteration, or use equal weight
            if len(weights_backtest.dropna(how='all')) > 0:
                # Get the last valid weights
                last_valid_weights = weights_backtest[asset_columns].iloc[-1]
                weights_old_filtered = np.array([
                    last_valid_weights[asset] if asset in last_valid_weights.index and not pd.isna(last_valid_weights[asset]) else 1/len(available_assets)
                    for asset in available_assets
                ])
                # Normalize to sum to 1
                if weights_old_filtered.sum() > 0:
                    weights_old_filtered = weights_old_filtered / weights_old_filtered.sum()
                else:
                    weights_old_filtered = np.array([1/len(available_assets) for _ in available_assets])
            else:
                weights_old_filtered = np.array([1/len(available_assets) for _ in available_assets])
            
            # Check if covariance matrix is positive definite
            eigenvalues = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvalues <= 0):
                warnings.warn(f"Skipping {ind}: Covariance matrix is not positive definite - using MonteCarlo")
                opt_weights = MonteCarlo_portfolio(1000, 
                                                    expected_returns_series, 
                                                    cov_matrix, 
                                                    strategy="Lowest std")
                performance = portfolio_performance(opt_weights, expected_returns_series, cov_matrix)
                
                weights_backtest.loc[ind, asset_columns] = 0.0
                for i, asset in enumerate(available_assets):
                    weights_backtest.loc[ind, asset] = opt_weights[i]
                
                weights_backtest.loc[ind, 'sharpe_ratio'] = sharpe_ratio(performance['return'],
                                                                         performance['std'],
                                                                         risk_free_rate)
                weights_backtest.loc[ind, 'expected_return'] = performance['return']
                weights_backtest.loc[ind, 'std'] = performance['std']
                continue
                
        except Exception as e:
            warnings.warn(f"Skipping {ind}: Data validation failed - {str(e)} (Available data rows: {len(return_tmp)}, Assets: {len(return_tmp.columns)}) - using previous weights")
            # Use previous weights or equal weights
            if len(weights_backtest.dropna(how='all')) > 0:
                last_weights = weights_backtest[asset_columns].iloc[-1]
                weights_backtest.loc[ind, asset_columns] = last_weights
            else:
                weights_backtest.loc[ind, asset_columns] = 1/len(asset_columns)
            weights_backtest.loc[ind, 'sharpe_ratio'] = np.nan
            weights_backtest.loc[ind, 'expected_return'] = np.nan
            weights_backtest.loc[ind, 'std'] = np.nan
            continue
        
        # Use L2 algorithm for optimization
        opt_weights = l2_algorithm(expected_returns_series,
                                   cov_matrix,
                                   weights_old_filtered,
                                   rho,
                                   gamma,
                                   max_weight)
        
        if opt_weights is None:
            warnings.warn(f"Skipping {ind}: Optimization failed to converge - using previous weights")
            # Use previous weights or equal weights
            if len(weights_backtest.dropna(how='all')) > 0:
                last_weights = weights_backtest[asset_columns].iloc[-1]
                weights_backtest.loc[ind, asset_columns] = last_weights
            else:
                weights_backtest.loc[ind, asset_columns] = 1/len(asset_columns)
            weights_backtest.loc[ind, 'sharpe_ratio'] = np.nan
            weights_backtest.loc[ind, 'expected_return'] = np.nan
            weights_backtest.loc[ind, 'std'] = np.nan
            continue
        
        # Calculate performance metrics
        performance = portfolio_performance(opt_weights, expected_returns_series, cov_matrix)
        sharpe = sharpe_ratio(performance['return'], performance['std'], risk_free_rate)
        
        # Store weights and metrics
        weights_backtest.loc[ind, asset_columns] = 0.0
        for i, asset in enumerate(available_assets):
            weights_backtest.loc[ind, asset] = opt_weights[i]
        
        weights_backtest.loc[ind, 'sharpe_ratio'] = sharpe
        weights_backtest.loc[ind, 'expected_return'] = performance['return']
        weights_backtest.loc[ind, 'std'] = performance['std']
    
    return weights_backtest
