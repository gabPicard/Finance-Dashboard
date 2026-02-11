import numpy as np
import warnings

def validate_and_clean_data(returns, min_observations=100, min_variance=1e-6, max_nan_percentage=0.2):
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