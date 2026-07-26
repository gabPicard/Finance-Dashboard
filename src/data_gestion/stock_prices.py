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
    Get clean and validate historical stock prices, market prices, and the risk free rate.
    
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

    :returns clean_prices: DataFrame of cleaned stock prices
    :rtype clean_prices: pd.DataFrame
    :returns clean_market_prices: DataFrame of cleaned market prices
    :rtype clean_market_prices: pd.DataFrame
    :returns risk_free_rate: The risk free return rate
    :rtype risk_free_rate: float
    :returns actual_tickers: List of tickers that remain after cleaning (may differ from initial list)
    :rtype actual_tickers: list[str]
    """
    
    if isinstance(market, list):
        tickers_list = merge_markets(market)
        risk_free_rate_ticker = "^IRX"
        market_ticker = "^GSPC"  # Default to S&P 500 for multiple markets
    else:
        tickers_list, risk_free_rate_ticker, market_ticker = get_tickers_list(market)

    raw_prices = fetch_stock_data(tickers_list,
                                  start_date=start_date,
                                  end_date=end_date,
                                  period=period,
                                  interval=interval)
    
    # Fetch market prices
    raw_market_prices = fetch_stock_data([market_ticker],
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
    
    # Process market prices
    market_prices_tmp = raw_market_prices[columns]
    
    if isinstance(market_prices_tmp.columns, pd.MultiIndex):
        market_prices_tmp.columns = [col[1] if col[1] else col[0] for col in market_prices_tmp.columns]
    
    market_prices_tmp, _ = fix_price_anomalies(market_prices_tmp, max_daily_change=0.5, max_anomalies=3)
    
    clean_market_prices, _ = validate_and_clean_data(market_prices_tmp)
    
    market_prices_reset = clean_market_prices.reset_index()
    if market_prices_reset.columns[0] == 'index' or market_prices_reset.columns[0] == 'Date':
        market_prices_reset = market_prices_reset.rename(columns={market_prices_reset.columns[0]: 'date'})
    else:
        market_prices_reset.insert(0, 'date', clean_market_prices.index)
    
    clean_market_prices = market_prices_reset
    
    return clean_prices, clean_market_prices, risk_free_rate


def get_tickers_list(market: str):
    """
    Get the list of every asset's ticker in the market and the risk free rate ticker
    
    :param market: Must be the name of a known market in the json file.
    :type market: str

    :returns tuple: tickers list | risk free rate | market ticker
    """
    json_path = os.path.join(os.path.dirname(__file__), 'tickers_list.json')
    with open(json_path) as f:
        data = json.load(f)
        tickers = data[market]['Tickers list']
        rfr = data[market]['Risk free rate']
        market_ticker = data[market]['Market ticker']
    
    if isinstance(rfr, list):
        rfr = rfr[0]
    
    return tickers, rfr, market_ticker

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
            tickers_list, rfr, market_ticker = get_tickers_list(market)
            for ticker in tickers_list:
                if ticker not in merged_list:
                    merged_list.append(ticker)
        except Exception as e:
            print(f"Error when fetching the list of tickers of {market}: {e}")
    return merged_list