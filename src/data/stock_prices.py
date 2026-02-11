import numpy as np
import pandas as pd
import json
import os
from .data_engineering import validate_and_clean_data
from .fetch_data import fetch_stock_data, fetch_risk_free_rate

def get_stock_prices(market, 
                     start_date=None, 
                     end_date=None, 
                     period="1y", 
                     interval="1d",
                     columns=['Close']):
    
    tickers_list, risk_free_rate_ticker = get_tickers_list(market)

    raw_prices = fetch_stock_data(tickers_list,
                                  start_date=start_date,
                                  end_date=end_date,
                                  period=period,
                                  interval=interval)
    risk_free_rate = fetch_risk_free_rate(risk_free_rate_ticker)

    prices_tmp = raw_prices[columns]

    clean_prices, excluded_assets = validate_and_clean_data(prices_tmp)

    return clean_prices, risk_free_rate


def get_tickers_list(market):
    json_path = os.path.join(os.path.dirname(__file__), 'tickers_list.json')
    with open(json_path) as f:
        data = json.load(f)
        tickers = data[market]['Tickers list']
        rfr = data[market]['Risk free rate']
    
    if isinstance(rfr, list):
        rfr = rfr[0]
    
    return tickers, rfr