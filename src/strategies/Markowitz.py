import numpy as np
import pandas as pd
import warnings
from qpsolvers import solve_qp
from ..metrics.portfolio_measurements import compound_growth_rate
from .CAPM import capm_expected_returns

def optimize_portfolio(cov_matrix: pd.DataFrame, 
                       expected_returns: pd.Series, 
                       target_return: float=None, 
                       max_weight: float=0.15
                    ) -> np.ndarray:
    """
    Optimize the weights of the portfolio to obtain the lowest Std, with a specific return or not.

    :param cov_matrix: The matrix of covariances bewteen assets
    :type cov_matrix: pd.Series
    :param expected_returns: The average return of the assets
    :type expected_returns: pd.DataFrame
    :param target_return: [Optional] The return we want to optimize for
    :type target_return: float
    :param max_weight: [Optional] The maximum weight an asset can hold in the portfolio
    :type max_weight: float

    :returns weights: A vector of each asset's weight in the portfolio
    :rtype weights: ndarray
    """
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values

    # Ensure max_weight is valid (between 1/n_assets and 1.0)
    if max_weight is not None:
        max_weight = max(min(max_weight, 1.0), 1/len(expected_returns))

    P = cov_matrix
    q = np.zeros(len(expected_returns))
    
    A_list = [np.ones((1, len(expected_returns)))]
    b_list = [1.0]
    
    if target_return is not None:
        A_list.append(expected_returns.reshape(1, -1))
        b_list.append(target_return)
    
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

def calculate_efficient_frontier(cov_matrix: pd.DataFrame, 
                                 expected_returns: pd.Series, 
                                 num_portfolios: int=100, 
                                 max_weight: float=0.15
                                 ) -> dict:
    """
    Compute an efficient frontier of optimized portfolios with different expected returns

    :param cov_matrix: the covariance matrix of all assets
    :type cov_matrix: pd.DataFrame
    :param expected_returns: the average return of all assets
    :type expected_returns: pd.Series
    :param num_portfolios: [Optional] The number of portfolios computed in the frontier
    :type num_portfolios: int
    :param max_weight: [Optional] The maximum weight an asset can hold in a portfolio
    :type max_weight: float

    :returns dict:
        - weights: np.array of list of each portfolio's weights

        - returns: np.array of each portfolio's expected return

        - std: list of each portfolio's std
    """
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

def portfolio_performance(weights: np.array, 
                          expected_returns: np.array, 
                          cov_matrix: np.ndarray
                        ) -> dict:
    """
    Compute all metrics to measure a portfolio performance

    :param weights: The assets' weights of the portfolio
    :type weights: np.array
    :param expected_returns: The assets' average return
    :type expected_returns: np.array
    :param cov_matrix: The matrix of covariances between assets
    :type cov_matrix: np.ndarray

    :returns dict: 
        - return
        
        - std
    """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    return {"return":portfolio_return, "std":portfolio_std}

def sharpe_ratio(portfolio_return: float, 
                 std: float, 
                 risk_free_rate: float
                 ) -> float:
    """
    Compute the Sharpe Ratio of a portfolio.

    :param portfolio_return: The expected return of the portfolio
    :type portfolio_return: float
    :param std: The Standard Deviation of the portfolio
    :type std: float
    :param risk_free_rate: The risk free return rate
    :type risk_free_rate: float

    :returns sharpe: The sharpe ratio
    """
    if std == 0 or np.isnan(std):
        return 0.0
    return (portfolio_return - risk_free_rate) / std

def best_sharpe_ratio(efficient_frontier: dict, 
                      risk_free_rate: float
                    ) -> dict:
    """
    Find the portfolio with the best Sharpe Ratio in an efficient frontier

    :param efficient_frontier: A dictionnary with all the portfolios' weight, return and std. 
                                Use the calculate_efficient_frontier function.
    :type efficient_frontier: dict
    :param risk_free_rate: The risk free return rate
    :type risk_free_rate: float

    :returns portfolio:
        - sharpe ratio

        - weights

        - expected return

        - std

    :rtype portfolio: dict
    """
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

def rolling_window(prices: pd.DataFrame, 
                   risk_free_rate: float, 
                   rebalance_frequency: str='QE', 
                   strategy: str="Best sharpe", 
                   max_weight: float=0.15
                ) -> pd.DataFrame:
    """
    Use the optimization algorithm at every rebalancing date on a long period of time.

    At each rebalancing date, the algorithm uses the past year data.

    :param prices: The assets prices
    :type prices: pd.DataFrame
    :param risk_free_rate: The risk-free return rate
    :type risk_free_rate: float
    :param rebalance_frequency: [Optional] The frequency of weights rebalancement. Must use a yfinance accepted std
    :type reblance_frequency: str
    :param strategy: [Optional] Wich strategy to use for weights: Lowest std or Best Sharpe Ratio ?
    :type strategy: str
    :param max_weight: [Optional] The maximum weight an asset can hold in the portfolio
    :type max_weight: float

    :returns weights: A DataFrame containing the portfolio's weights with every rebalancing date as the index
    :rtype weights: pd.DataFrame

    """
    prices_copy = prices.copy()
    prices_copy['date'] = pd.to_datetime(prices_copy['date'])
    prices_copy = prices_copy.set_index('date').sort_index()
    
    index_rebalancement = prices_copy.resample(rebalance_frequency).last().index
    
    asset_columns = list(prices_copy.columns)
    metric_columns = ['sharpe_ratio', 'expected_return', 'std']
    all_columns = asset_columns + metric_columns

    # Ensure max_weight is valid (between 1/n_assets and 1.0)
    if max_weight is not None:
        max_weight = max(min(max_weight, 1.0), 1/len(asset_columns))
    
    weights_backtest = pd.DataFrame(index=index_rebalancement, columns=all_columns)
    last_valid_weights = None
    
    for ind in index_rebalancement:
        price_tmp = prices_copy[:ind].tail(252)
        
        # Skip if insufficient data (need at least 252 days for annual calculations)
        if len(price_tmp) < 252:
            warnings.warn(f"Skipping {ind}: Insufficient data ({len(price_tmp)} days, need 252)")
            continue
        
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
                
                last_valid_weights = {
                    'weights': opt_weights,
                    'sharpe_ratio': sharpe,
                    'expected_return': performance['return'],
                    'std': performance['std']
                }

    return weights_backtest

def l2_algorithm(expected_returns: np.array,
                 cov_matrix: np.array,
                 old_weights: np.array,
                 rho: float,
                 gamma: float,
                 max_weight: float
                ) -> np.array:
    """
    The l2 Algorithm to optimize a given portfolio using learning parameters and past results.

    :param expected_returns: The assets's expected returns
    :type expected_returns: np.array
    :param cov_matrix: The covariance matrix
    :type cov_matrix: np.array
    :param old_weights: The past optimized weights
    :type old_weights: np.array
    :param rho: Learning parameter rho
    :type rho: float
    :param gamma: Learning parameter gamma
    :type gamma: float
    :param max_weight: The maximum weight a single asset can have
    :type max_weight: float

    :returns opt_weights: The new optimized weights
    :rtype opt_weights: np.array
    """

    n = len(expected_returns)

    P = cov_matrix
    q = np.zeros(n)

    A_list = [np.ones((1, n))]
    b_list = [1.0]

    A = np.vstack(A_list)
    b = np.array(b_list)

    G_list = []
    h_list = []
    G_list.append(-np.eye(n))
    h_list.append(np.zeros(n))

    if max_weight is not None:
        G_list.append(np.eye(n))
        h_list.append(np.full(n, max_weight))

    if G_list:
        G = np.vstack(G_list)
        h = np.concatenate(h_list)
    else:
        G = None
        h = None
    
    In = np.eye((n))
    P += 2 * rho * In
    q = -gamma * expected_returns + 2 * rho * old_weights.T

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        opt_weights = solve_qp(P, q, G, h, A, b, solver="ecos")
    
    return opt_weights


def l2_optimization(prices: pd.DataFrame, 
                    risk_free_rate: float, 
                    rho: float, 
                    gamma: float, 
                    rebalance_frequency: str='QE', 
                    max_weight: float=0.15,
                    computation_function: str="cgr"
                ) -> pd.DataFrame:
    """
    Use the optimization algorithm at every rebalancing date on a long period of time, using an L2-optimization algorithm.

    At each rebalancing date, the algorithm uses the past year data.

    :param prices: The assets prices
    :type prices: pd.DataFrame
    :param risk_free_rate: The risk-free return rate
    :type risk_free_rate: float
    :param rho: Represents how much the past portfolio matters in the current one
    :type rho: float
    :param gamma: L2-optimization parameter
    :type gamma: float
    :param rebalance_frequency: [Optional] The frequency of weights rebalancement. Must use a yfinance accepted std
    :type reblance_frequency: str
    :param max_weight: [Optional] The maximum weight an asset can hold in the portfolio
    :type max_weight: float

    :returns weights: A DataFrame containing the portfolio's weights with every rebalancing date as the index
    :rtype weights: pd.DataFrame
    """
    prices_copy = prices.copy()
    prices_copy['date'] = pd.to_datetime(prices_copy['date'])
    prices_copy = prices_copy.set_index('date').sort_index()
    
    index_rebalancement = prices_copy.resample(rebalance_frequency).last().index
    
    asset_columns = list(prices_copy.columns)
    metric_columns = ['sharpe_ratio', 'expected_return', 'std']
    all_columns = asset_columns + metric_columns

    if max_weight is not None:
        max_weight = max(min(max_weight, 1.0), 1/len(asset_columns))
    
    weights_backtest = pd.DataFrame(index=index_rebalancement, columns=all_columns)
    weights_old = np.array([1/len(asset_columns) for _ in range(len(asset_columns))])
    
    for ind in index_rebalancement:
        price_tmp = prices_copy[:ind].tail(252)
        
        if len(price_tmp) < 63:
            warnings.warn(f"Skipping {ind}: Insufficient data ({len(price_tmp)} days, need 63)")
            continue
        
        return_tmp = price_tmp.pct_change(fill_method=None).dropna()
        
        try:
            
            cov_matrix = return_tmp.cov() * 252

            match computation_function:
                case "cgr":
                    expected_returns = compound_growth_rate(price_tmp, 252)
                case _:
                    expected_returns = np.mean(return_tmp)
            
            eigenvalues = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvalues <= 0):
                warnings.warn(f"Skipping {ind}: Covariance matrix is not positive definite - using MonteCarlo")
                opt_weights = MonteCarlo_portfolio(1000, 
                                                    expected_returns, 
                                                    cov_matrix, 
                                                    strategy="Lowest std")
                performance = portfolio_performance(opt_weights, expected_returns, cov_matrix)
                
                weights_backtest.loc[ind, asset_columns] = 0.0
                for i, asset in enumerate(return_tmp.columns):
                    weights_backtest.loc[ind, asset] = opt_weights[i]
                
                weights_backtest.loc[ind, 'sharpe_ratio'] = sharpe_ratio(performance['return'],
                                                                         performance['std'],
                                                                         risk_free_rate)
                weights_backtest.loc[ind, 'expected_return'] = performance['return']
                weights_backtest.loc[ind, 'std'] = performance['std']
                
                weights_old = opt_weights
                continue
                
        except Exception as e:
            warnings.warn(f"Skipping {ind}: Data validation failed - {str(e)} (Available data rows: {len(return_tmp)}, Assets: {len(return_tmp.columns)}) - using previous weights")
            weights_backtest.loc[ind, asset_columns] = weights_old
            weights_backtest.loc[ind, 'sharpe_ratio'] = np.nan
            weights_backtest.loc[ind, 'expected_return'] = np.nan
            weights_backtest.loc[ind, 'std'] = np.nan
            continue

        opt_weights = l2_algorithm(expected_returns,
                                   cov_matrix,
                                   weights_old,
                                   rho,
                                   gamma,
                                   max_weight)
        
        if opt_weights is None:
            warnings.warn(f"Skipping {ind}: Optimization failed to converge - using previous weights")
            weights_backtest.loc[ind, asset_columns] = weights_old
            weights_backtest.loc[ind, 'sharpe_ratio'] = np.nan
            weights_backtest.loc[ind, 'expected_return'] = np.nan
            weights_backtest.loc[ind, 'std'] = np.nan
            continue
        
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

def find_best_params(prices: pd.DataFrame, 
                     risk_free_rate: float, 
                     deciding_value: str="std"
                    ) -> tuple:
    """
    Function to find the best L2-optimization parameter based on a comparison variable

    :param prices: The assets' historical prices
    :type prices: pd.DataFrame
    :param risk_free_rate: The risk-free return rate
    :type risk_free_rate: float
    :param deciding_value: [Optional] The value used to compare the optimized portfolios
    :type deciding_value: str

    :returns tuple:
        - Best rho

        - Best gamma
    
    :rtype tuple: tuple
    """
    rebalance_frequency = 'QE'
    max_weight = 0.15

    gammas = np.linspace(0.01, 0.05, 10)
    rhos = np.linspace(0.01, 1, 100)

    metrics = ['Return', 'Sharpe Ratio', 'Standard Deviation']
    params = pd.DataFrame(
        index=pd.Index(gammas, name="gamma"),
        columns=pd.MultiIndex.from_product([rhos, metrics], names=["rho", "metric"]),
    )

    max_metric = 0
    best_gamma = 0
    best_rho = 0

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

            if deciding_value == "return":
                if last_metrics['expected_return'] > max_metric:
                    max_metric = last_metrics['expected_return']
                    best_gamma, best_rho = ind_gammas, ind_rhos
            elif deciding_value == "std":
                if -last_metrics['std'] < max_metric:
                    max_metric = -last_metrics['std']
                    best_gamma, best_rho = ind_gammas, ind_rhos
            elif deciding_value == "sharpe":
                if last_metrics['sharpe_ratio'] > max_metric:
                    max_metric = last_metrics['sharpe_ratio']
                    best_gamma, best_rho = ind_gammas, ind_rhos
    
    if deciding_value is None:
        return params

    return best_gamma, best_rho

def MonteCarlo_portfolio(precision: int,
                         expected_returns: np.ndarray,
                         cov_matrix: np.ndarray,
                         risk_free_rate: float = 0.04,
                         strategy: str = "Lowest std"
                        ) -> np.ndarray:
    """
    Use MonteCarlo method to get a portfolio, with either the lowest std or the best sharpe.
    Less efficient than the regular optimization, but doesn't rely on matrixes being defined positive.

    :param precision: The number of portfolio created at random
    :type precision: int
    :param expected_returns: The expected_returns of each assets
    :type expected_returns: np.ndarray
    :param cov_matrix: The covariance matrix between assets
    :type cov_matrix: np.ndarray
    :param risk_free_rate: [Optional] The risk free return rate. By default, 4%
    :type risk_free_rate: float
    :param strategy: [Optional] The metrics used to get the portfolio. Either "Lowest std" or "Best sharpe"
    :type strategy: str

    :returns weights: Assets' weight
    :rtype weights: np.ndarray
    """
    best_weights = np.array([])
    best_param = 0 if strategy == "Best sharpe" else float('inf')
    
    for i in range(precision):
        random_weights = np.random.dirichlet(np.ones(expected_returns.shape[0]))
        performance = portfolio_performance(random_weights, expected_returns, cov_matrix)
        rtrn = performance['return']
        std = performance['std']
        
        if strategy == "Best sharpe":
            sharpe = sharpe_ratio(rtrn, std, risk_free_rate)
            if sharpe > best_param:
                best_param = sharpe
                best_weights = random_weights
        else:
            if std < best_param:
                best_param = std
                best_weights = random_weights
    
    return best_weights