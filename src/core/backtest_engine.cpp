/**
 * backtest_engine.cpp
 *
 * Bar-by-bar portfolio backtesting loop.
 *
 * For each day t:
 *   1. Identify the active weight vector (most recent rebalancing <= t).
 *   2. Compute the portfolio return as the weighted sum of asset returns.
 *
 * Returns are computed as simple (arithmetic) returns: r_t = P_t/P_{t-1} − 1.
 */

#include "backtest_engine.h"
#include <stdexcept>

namespace finance {

Eigen::VectorXd run_backtest(
    const Eigen::MatrixXd& price_matrix,
    const Eigen::MatrixXd& weight_matrix,
    const std::vector<int>& rebalance_days)
{
    int T = static_cast<int>(price_matrix.rows());
    int N = static_cast<int>(price_matrix.cols());
    int K = static_cast<int>(weight_matrix.rows());

    if (static_cast<int>(rebalance_days.size()) != K) {
        throw std::invalid_argument(
            "rebalance_days length must match the number of rows in weight_matrix");
    }
    if (weight_matrix.cols() != N) {
        throw std::invalid_argument(
            "weight_matrix and price_matrix must have the same number of columns");
    }
    if (T < 2) {
        throw std::invalid_argument("price_matrix must have at least 2 rows");
    }

    Eigen::VectorXd portfolio_returns = Eigen::VectorXd::Zero(T);
    // Day 0 has no previous price, so return is 0.

    int current_rebal_idx = 0;
    Eigen::VectorXd active_weights = weight_matrix.row(0);

    for (int t = 1; t < T; ++t) {
        // Advance to the latest rebalancing that has occurred on or before day t
        while (current_rebal_idx + 1 < K &&
               rebalance_days[current_rebal_idx + 1] <= t)
        {
            ++current_rebal_idx;
            active_weights = weight_matrix.row(current_rebal_idx);
        }

        // Compute asset returns: r_i = P_{t,i} / P_{t-1,i} − 1
        Eigen::VectorXd asset_returns(N);
        for (int i = 0; i < N; ++i) {
            double prev = price_matrix(t - 1, i);
            if (prev != 0.0) {
                asset_returns(i) = price_matrix(t, i) / prev - 1.0;
            } else {
                asset_returns(i) = 0.0;
            }
        }

        portfolio_returns(t) = active_weights.dot(asset_returns);
    }

    return portfolio_returns;
}

} // namespace finance
