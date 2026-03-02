import pandas as pd
import json
import os
from .data_engineering import validate_and_clean_data, fix_price_anomalies, delete_assets
from .fetch_data import fetch_stock_data, fetch_risk_free_rate

def get_stock_prices(market: str | list[str], 
                     start_date=None, 
                     end_date=None, 
                     period="1y", 
                     interval="1d",
                     columns=['Close']
                    ) -> tuple:
    """
    Get clean and validate historical stock prices and the risk free rate.
    
    :param market: The name of a market, or a list of markets. If it is a list, the program will get each tickers from each markets
    :type market: str | list[str]
    :param start_date: [Optional] The starting point for fetching. If None, will use period to compute the start date
    :type start_date: str
    :param end_date: [Optional] The end point for fetching. By default, today
    :type end_date: str
    :param period: [Optional] The period for fetching. By default, 1 year
    :type period: str
    :param interval: [Optional] The interval to fecth data. By default, 1 day
    :type interval: str

    :returns clean_prices: DataFrame of cleaned prices
    :rtype clean_prices: pd.DataFrame
    :returns risk_free_rate: The risk free return rate
    :rtype risk_free_rate: float
    :returns actual_tickers: List of tickers that remain after cleaning (may differ from initial list)
    :rtype actual_tickers: list[str]
    """
    
    if isinstance(market, list):
        tickers_list = merge_markets(market)
        risk_free_rate = "^IRX"
    else:
        tickers_list, risk_free_rate_ticker = get_tickers_list(market)

    raw_prices = fetch_stock_data(tickers_list,
                                  start_date=start_date,
                                  end_date=end_date,
                                  period=period,
                                  interval=interval)
    risk_free_rate = fetch_risk_free_rate(risk_free_rate_ticker)

    prices_tmp = raw_prices[columns]

    if isinstance(prices_tmp.columns, pd.MultiIndex):
        prices_tmp.columns = [col[1] if col[1] else col[0] for col in prices_tmp.columns]

    prices_tmp, excluded_anomalies = fix_price_anomalies(prices_tmp, max_daily_change=0.5, max_anomalies=3)

    clean_prices, excluded_assets = validate_and_clean_data(prices_tmp)

    prices_reset = clean_prices.reset_index()
    if prices_reset.columns[0] == 'index' or prices_reset.columns[0] == 'Date':
        prices_reset = prices_reset.rename(columns={prices_reset.columns[0]: 'date'})
    else:
        prices_reset.insert(0, 'date', clean_prices.index)

    clean_prices = prices_reset

    all_excluded = excluded_anomalies + [ticker for ticker, reason in excluded_assets]
    if all_excluded:
        delete_assets(all_excluded, market)
    
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

def merge_markets(market_list: list[str]) -> list[str]:
    """
    Create a single list of of tickers from multiple markets. Each ticker appear only once.

    :param market_list: The list of all markets
    :type market_list: list[str]

    :returns tickers_list: A single list of all tickers
    :rtype tickers_list: list[str]
    """
    merged_list = []
    for market in market_list:
        try:
            tickers_list, rfr = get_tickers_list(market)
            for ticker in tickers_list:
                if ticker not in merged_list:
                    merged_list.append(ticker)
        except Exception as e:
            print(f"Error when fetching the list of tickers of {market}: {e}")
    return merged_list