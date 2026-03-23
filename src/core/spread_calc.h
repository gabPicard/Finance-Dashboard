#pragma once
/**
 * spread_calc.h
 *
 * Rolling z-score computation on a spread series.
 *
 * For each position t, the z-score is:
 *   z_t = (x_t − mean(x_{t-window+1..t})) / std(x_{t-window+1..t})
 *
 * Positions where the rolling window is not yet full return NaN.
 */

#include <Eigen/Dense>

namespace finance {

/**
 * @brief Compute the rolling z-score of a 1-D spread series.
 *
 * @param spread  Input spread vector of length T.
 * @param window  Rolling window size.
 * @return Eigen::VectorXd  Z-score vector of length T.  The first (window − 1)
 *         entries are set to NaN.
 */
Eigen::VectorXd rolling_zscore(const Eigen::VectorXd& spread, int window);

} // namespace finance
