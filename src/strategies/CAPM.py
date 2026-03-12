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

def capm_expected_returns(prices: pd.DataFrame,
                          market_prices: pd.DataFrame,
                          risk_free_rate: float=0.04
                        ) -> pd.Series:
   """
   Compute the expected returns of assets using the CAPM

   :param prices: The assets' historical prices
   :type prices: pd.DataFrame
   :param market_prices: The market historical prices
   :type market_prices: pd.DatFrame
   :param risk_free_rate: The risk-free return rate. By default 4%
   :type risk_free_rate: float

   :returns capm_returns: dict{asset: capm_expected_return}
   :rtype capm_returns: dict
   """
   market_returns = market_prices.pct_change(fill_method=None).dropna()
   asset_returns = prices.pct_change(fill_method=None).dropna()
   
   # Extract the first column if market_returns is a DataFrame
   if isinstance(market_returns, pd.DataFrame):
      market_returns = market_returns.iloc[:, 0]

   # Ensure market_expected_return is a scalar
   market_expected_return = float(market_returns.mean() * 252)

   capm_returns = {}
   for asset in asset_returns.columns:
      beta_i = calculate_beta(asset_returns[asset], market_returns)
      capm_returns[asset] = float(risk_free_rate + beta_i * (market_expected_return - risk_free_rate))
   
   return capm_returns