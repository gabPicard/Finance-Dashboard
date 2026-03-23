#pragma once
/**
 * backtest_engine.h
 *
 * Fast bar-by-bar backtesting loop.
 *
 * Given a price matrix P (T × N) and a weight matrix W (K × N) where K <= T
 * is the number of rebalancing events, this module computes the daily portfolio
 * returns vector of length T.
 *
 * Rebalancing dates are provided as a vector of integer row indices into P.
 */

#include <Eigen/Dense>
#include <vector>

namespace finance {

/**
 * @brief Run a bar-by-bar portfolio backtest.
 *
 * @param price_matrix    (T × N) matrix of asset prices (rows=days, cols=assets).
 * @param weight_matrix   (K × N) matrix of portfolio weights at each rebalancing.
 * @param rebalance_days  Sorted vector of row indices (into price_matrix) at which
 *                        weights are updated.  Length must equal K.
 * @return Eigen::VectorXd  Portfolio return at each of the T bars (daily returns).
 */
Eigen::VectorXd run_backtest(
    const Eigen::MatrixXd& price_matrix,
    const Eigen::MatrixXd& weight_matrix,
    const std::vector<int>& rebalance_days);

} // namespace finance
