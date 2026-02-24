import numpy as np
import warnings
import pandas as pd
import json
from .fetch_data import get_company_name

def fix_price_anomalies(prices: pd.DataFrame, max_daily_change: float = 0.5, max_anomalies: int = 3) -> tuple:
    """
    Detect and fix assets with extreme price jumps by replacing anomalous values with previous day's price.
    Assets with too many anomalies are flagged for removal.
    
    :params prices: DataFrame of prices with assets as columns
    :type prices: pd.DataFrame
    :params max_daily_change: Maximum acceptable daily price change (default 50%)
    :type max_daily_change: float
    :params max_anomalies: Maximum number of anomalies before flagging asset for removal (default 3)
    :type max_anomalies: int
    :returns: Tuple of (corrected prices DataFrame, list of assets to exclude)
    """
    prices_corrected = prices.copy()
    
    price_changes = prices.pct_change(fill_method=None)
    
    total_fixes = 0
    anomaly_counts = {}
    assets_to_exclude = []
    
    for col in prices.columns:
        extreme_changes = price_changes[col][abs(price_changes[col]) > max_daily_change]
        anomaly_counts[col] = len(extreme_changes)
        
        if len(extreme_changes) > 0:
            if len(extreme_changes) > max_anomalies:
                assets_to_exclude.append(col)
                warnings.warn(
                    f"Asset {col} has {len(extreme_changes)} anomalies (>{max_anomalies}), will be excluded"
                )
                continue
            
            for date, change in extreme_changes.items():
                idx = prices.index.get_loc(date)
                if idx > 0:
                    prev_date = prices.index[idx - 1]
                    prev_price = prices.loc[prev_date, col]
                    original_price = prices.loc[date, col]
                    
                    prices_corrected.loc[date, col] = prev_price
                    total_fixes += 1
                    
                    warnings.warn(
                        f"Fixed extreme price change for {col} on {date.date() if hasattr(date, 'date') else date}: "
                        f"{change:+.2%} (${original_price:.2f} → ${prev_price:.2f})"
                    )
    
    if total_fixes > 0:
        print(f"Fixed {total_fixes} anomalous price point(s) across {len(prices.columns)} assets")
    
    if assets_to_exclude:
        print(f"Excluding {len(assets_to_exclude)} asset(s) due to excessive anomalies: {assets_to_exclude}")
        prices_corrected = prices_corrected.drop(columns=assets_to_exclude)
    
    return prices_corrected, assets_to_exclude

def validate_and_clean_data(returns: np.ndarray, 
                            min_observations: int =100, 
                            min_variance: float =1e-6, 
                            max_nan_percentage: float =0.2) -> np.ndarray | list[str]:
    """
    Validate and clean returns data by removing assets with excessive NaN values, near-zero variance. 
    
    :params returns: Dataframe of asset returns with assets as columns
    :type returns: np.ndarray
    
    :returns tuple: cleaned_returns DataFrame, list of (asset, reason)
    """
    excluded_assets = []
    valid_columns = []
    
    for col in returns.columns:
        series = returns[col]
        
        nan_count = series.isna().sum()
        if nan_count > len(series) * max_nan_percentage:
            excluded_assets.append((col, f"Too many NaN values ({nan_count}/{len(series)})"))
            continue
            
        valid_obs = series.dropna()
        if len(valid_obs) < min_observations:
            excluded_assets.append((col, f"Insufficient observations ({len(valid_obs)} < {min_observations})"))
            continue
            
        variance = valid_obs.var()
        if variance < min_variance or np.isnan(variance):
            excluded_assets.append((col, f"Zero or near-zero variance ({variance})"))
            continue
            
        if valid_obs.nunique() <= 1:
            excluded_assets.append((col, "Constant values"))
            continue
            
        valid_columns.append(col)
    
    if len(valid_columns) == 0:
        raise ValueError("No valid assets remaining after data validation")
    
    cleaned_returns = returns[valid_columns].copy()
    
    cleaned_returns = cleaned_returns.ffill().bfill()
    
    cleaned_returns = cleaned_returns.dropna()
    
    if len(excluded_assets) > 0:
        warnings.warn(f"Excluded {len(excluded_assets)} assets due to data quality issues")
    
    return cleaned_returns, excluded_assets

def format_portfolio(weights: np.ndarray,
                     tickers_list: list[str],
                     portfolio_value: float = 100,
                     to_txt: bool = False,
                     txt_file_name: str = None
                    ) -> str:
    """
    Properly give a detailled assets repartition in the portfolio.

    :param weights: The weights of each assets, in order
    :type weights: np.ndarray
    :param tickers_list: Each asset's ticker, with the same order
    :type tickers_list: list[str]
    :param portfolio_value: [Optional] The total value of the portfolio, by default 100
    :type portfolio_value: float
    :param to_txt: Create a .txt file if this boolean is true
    :type to_txt: bool
    :param txt_file_name: The name of the file to save the assets' weights
    :type txt_file_name: str

    :returns repartition: assets' weights. None if it encounters an error
    :rtype repartition: str
    """
    if weights is None:
        print("Weights list is None")
        return None
    if tickers_list is None or len(tickers_list) == 0:
        print("Tickers list doesn't exist")
        return None
    if weights.shape[0] != len(tickers_list):
        print("Weights list and Tickers list must have the same format")
        return None
    if to_txt and txt_file_name is None:
        txt_file_name = "weights repartition"
    repartition = ""
    for i in range (0, weights.shape[0]):
        repartition += tickers_list[i] + "   "
        + get_company_name(tickers_list[i])
        + weights[i]*100 + "%   "
        + weights[i]*portfolio_value + "\n"
    if to_txt:
        with open(txt_file_name, "w") as file:
            file.write(repartition)
    return repartition

def delete_assets(excluded_assets: list[str], 
                  market: str
                ) -> None:
    """
    Delete assets from the ticker list in the json file for a specified market.

    :param excluded_assets: Assets to exclude, must be a list of tickers
    :type excluded_assets: list[str]
    :param market: The market we want to update
    :type market: str
    """
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'tickers_list.json')
        with open(json_path) as f:
            data = json.load(f)
            tickers = data[market]['Tickers list']
        if check_assets_in_market(excluded_assets, tickers):
            exclude_set = set(excluded_assets)
            tickers_updated = [t for t in tickers if t not in exclude_set]

            data[market]['Tickers list'] = tickers_updated

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            print("Assets to exclude are not present in this market")
    except Exception as e: 
        print(f"Error while modifying the json: {e}")

def check_assets_in_market(assets: list[str],
                           market_list: list[str]
                        ) -> bool:
    """
    Check if all assets tickers are in the market

    :param assets: The list of assets we want to check
    :type assets: list[str]
    :param market_list: The list of all tickers from a market
    :type market_list: list[str]

    :returns check: True if all assets are in the market, False otherwise
    :retype check: bool
    """
    check = True
    for a in assets:
        if a not in market_list:
            check = False
    return check