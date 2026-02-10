import yfinance as yf
import pandas as pd
from typing import List, Optional, Union


def fetch_stock_data(
    tickers: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]

    if start_date and end_date:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
    else:
        data = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)
    
    return data


def fetch_stock_info(tickers: Union[str, List[str]]) -> dict:
    if isinstance(tickers, str):
        tickers = [tickers]
    print(f"Fetching info for tickers: {tickers}")
    stock_info = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_info[ticker] = stock.info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            stock_info[ticker] = None
    
    return stock_info


def fetch_risk_free_rate(ticker: str = "^FVX") -> float:
    """
    - ^IRX : Bons du Trésor à 13 semaines
    - ^FVX : Bons du Trésor à 5 ans
    - ^TNX : Bons du Trésor à 10 ans (défaut)
    - ^TYX : Bons du Trésor à 30 ans
    
    Returns annual rate as percentage (e.g., 4.5 for 4.5%)
    """
    try:
        treasury = yf.Ticker(ticker)
        hist = treasury.history(period="5d")
        if not hist.empty:
            # Le taux est déjà en pourcentage, le convertir en décimal
            rate = hist['Close'].iloc[-1] / 100  # Convert percentage to decimal
            return rate
        else:
            print(f"Aucune donnée disponible pour {ticker}, utilisation du taux par défaut de 4.0%")
            return 0.04  # 4% as decimal
    except Exception as e:
        print(f"Erreur lors de la récupération du taux sans risque: {e}")
        print("Utilisation du taux par défaut de 4.0%")
        return 0.04  # 4% as decimal


def clean_price_data(data: pd.DataFrame, min_valid_ratio: float = 0.5) -> pd.DataFrame:
    """
    Remove columns (stocks) with too many missing values or constant prices.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data (usually data['Close'])
    min_valid_ratio : float
        Minimum ratio of valid (non-NaN) values required (default 0.5 = 50%)
    
    Returns:
    --------
    cleaned_data : pd.DataFrame
        Data with problematic stocks removed
    """
    if isinstance(data.columns, pd.MultiIndex):
        # Handle MultiIndex columns from yfinance
        data = data.droplevel(0, axis=1)
    
    removed_stocks = []
    valid_columns = []
    
    for col in data.columns:
        series = data[col]
        
        # Check for too many NaN values
        valid_ratio = series.notna().sum() / len(series)
        if valid_ratio < min_valid_ratio:
            removed_stocks.append((col, f"Only {valid_ratio:.1%} valid data"))
            continue
        
        # Check for constant or near-constant values
        non_null_series = series.dropna()
        if len(non_null_series) > 0 and non_null_series.std() < 1e-6:
            removed_stocks.append((col, "Constant/near-constant prices"))
            continue
        
        valid_columns.append(col)
    
    if removed_stocks:
        print(f"\n⚠️  Removed {len(removed_stocks)} problematic stocks:")
        for stock, reason in removed_stocks:
            print(f"   - {stock}: {reason}")
    
    cleaned_data = data[valid_columns]
    print(f"\n✓ {len(valid_columns)} valid stocks remaining")
    
    return cleaned_data
