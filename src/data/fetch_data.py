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