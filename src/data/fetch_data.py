import yfinance as yf
import pandas as pd


def fetch_stock_data(
    tickers: str | list[str],
    start_date: str = None,
    end_date: str = None,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical data about a stock or a list of stocks.

    :return: ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    :rtype: dict
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    if start_date and end_date:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
    else:
        data = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)
    
    return data


def fetch_stock_info(tickers: str | list[str]) -> dict:
    """
    Fetch numerous informations about a stock or a list of stocks
    """
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
    Fetch the risk free rate, returns 4% by default.
    """
    try:
        treasury = yf.Ticker(ticker)
        hist = treasury.history(period="5d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1]
            if rate > 1:
                rate = rate / 100
            if rate < 0 or rate > 0.20:
                return 0.04
            return rate
        else:
            return 0.04
    except Exception as e:
        print(f"Erreur lors de la récupération du taux sans risque: {e}")
        print("Utilisation du taux par défaut de 4.0%")
        return 0.04