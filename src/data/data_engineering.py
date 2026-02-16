import numpy as np
import warnings
import pandas as pd

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