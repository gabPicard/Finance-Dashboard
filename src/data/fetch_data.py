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
    
    Returns annual rate as decimal (e.g., 0.045 for 4.5%)
    """
    try:
        treasury = yf.Ticker(ticker)
        hist = treasury.history(period="5d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1]
            # If rate is already in percentage format (0-100), convert to decimal
            if rate > 1:
                rate = rate / 100
            # Sanity check: reasonable risk-free rates are between 0% and 20%
            if rate < 0 or rate > 0.20:
                print(f"Taux suspect ({rate*100:.2f}%), utilisation du taux par défaut de 4.0%")
                return 0.04
            return rate
        else:
            print(f"Aucune donnée disponible pour {ticker}, utilisation du taux par défaut de 4.0%")
            return 0.04  # 4% as decimal
    except Exception as e:
        print(f"Erreur lors de la récupération du taux sans risque: {e}")
        print("Utilisation du taux par défaut de 4.0%")
        return 0.04  # 4% as decimal