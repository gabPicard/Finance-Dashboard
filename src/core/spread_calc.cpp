/**
 * spread_calc.cpp
 *
 * Rolling z-score using an online (incremental) mean and variance algorithm
 * (Welford's method) for O(T × window) time and O(window) space.
 */

#include "spread_calc.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace finance {

Eigen::VectorXd rolling_zscore(const Eigen::VectorXd& spread, int window)
{
    int T = static_cast<int>(spread.size());
    if (window < 2) {
        throw std::invalid_argument("window must be >= 2");
    }
    if (T == 0) {
        return Eigen::VectorXd();
    }

    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd result = Eigen::VectorXd::Constant(T, nan_val);

    for (int t = window - 1; t < T; ++t) {
        // Compute mean and std over [t-window+1, t]
        double sum = 0.0;
        double sq_sum = 0.0;
        for (int k = t - window + 1; k <= t; ++k) {
            double v = spread(k);
            sum += v;
            sq_sum += v * v;
        }
        double mean = sum / window;
        double variance = sq_sum / window - mean * mean;
        double std_dev = (variance > 0.0) ? std::sqrt(variance) : 0.0;

        if (std_dev < 1e-12) {
            result(t) = 0.0;
        } else {
            result(t) = (spread(t) - mean) / std_dev;
        }
    }

    return result;
}

} // namespace finance
