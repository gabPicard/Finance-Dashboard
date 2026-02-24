import json
import os
from .data_engineering import validate_and_clean_data, fix_price_anomalies
from .fetch_data import fetch_stock_data, fetch_risk_free_rate

def get_stock_prices(market, 
                     start_date=None, 
                     end_date=None, 
                     period="1y", 
                     interval="1d",
                     columns=['Close']):
    """
    Get clean and validate historical stock prices and the risk free rate.
    
    :param market: Must be the name of a known market in the json file.
    :type market: str

    :returns clean_prices | risk free rate:
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

            data[market]['Tickers lis'] = tickers_updated

            with json_path.open("w", encoding="utf-8") as f:
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