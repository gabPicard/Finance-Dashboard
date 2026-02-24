import json
import os
from .data_engineering import validate_and_clean_data, fix_price_anomalies, delete_assets
from .fetch_data import fetch_stock_data, fetch_risk_free_rate

def get_stock_prices(market: str, 
                     start_date=None, 
                     end_date=None, 
                     period="1y", 
                     interval="1d",
                     columns=['Close']
                    ) -> dict | float:
    """
    Get clean and validate historical stock prices and the risk free rate.
    
    :param market: Must be the name of a known market in the json file.
    :type market: str
    :param start_date: [Optional] The starting point for fetching. If None, will use period to compute the start date
    :type start_date: str
    :param end_date: [Optional] The end point for fetching. By default, today
    :type end_date: str
    :param period: [Optional] The period for fetching. By default, 1 year
    :type period: str
    :param interval: [Optional] The interval to fecth data. By default, 1 day
    :type interval: str

    :returns clean_prices: ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    :rtype clean_prices: dict
    :returns risk_free_rate: The risk free return rate
    :rtype rirks_free_rate: float
    """
    
    tickers_list, risk_free_rate_ticker = get_tickers_list(market)

    raw_prices = fetch_stock_data(tickers_list,
                                  start_date=start_date,
                                  end_date=end_date,
                                  period=period,
                                  interval=interval)
    risk_free_rate = fetch_risk_free_rate(risk_free_rate_ticker)

    prices_tmp = raw_prices[columns]

    prices_tmp, excluded_anomalies = fix_price_anomalies(prices_tmp, max_daily_change=0.5, max_anomalies=3)

    clean_prices, excluded_assets = validate_and_clean_data(prices_tmp)

    if len(excluded_assets) > 0:
        delete_assets(excluded_assets, market)

    return clean_prices, risk_free_rate


def get_tickers_list(market: str):
    """
    Get the list of every asset's ticker in the market and the risk free rate ticker
    
    :param market: Must be the name of a known market in the json file.
    :type market: str

    :returns tuple: tickers list | risk free rate
    """
    json_path = os.path.join(os.path.dirname(__file__), 'tickers_list.json')
    with open(json_path) as f:
        data = json.load(f)
        tickers = data[market]['Tickers list']
        rfr = data[market]['Risk free rate']
    
    if isinstance(rfr, list):
        rfr = rfr[0]
    
    return tickers, rfr