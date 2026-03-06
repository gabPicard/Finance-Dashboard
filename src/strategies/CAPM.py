import numpy as np
import pandas as pd

def calculate_beta(asset_returns: pd.Series,
                   market_returns: pd.Series
                ) -> float:
    """
    Compute the Beta of the asset

    :param asset_returns: Historical returns of the asset
    :type asset_returns: pd.Series
    :param market_returns: Historical returns of the market
    :type market_returns: pd.Series

    :returns beta: The asset's exposure to the market
    :rtype beta: float
    """
    covariance = asset_returns.cov(market_returns)
    market_var = market_returns.var()

    beta = covariance/market_var

    return beta

