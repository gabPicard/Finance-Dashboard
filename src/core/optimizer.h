#pragma once
/**
 * optimizer.h
 *
 * Markowitz quadratic-programming solver using Eigen.
 *
 * Solves: min  w^T Σ w  −  risk_aversion × μ^T w
 *         s.t. sum(w) = 1,  w >= 0
 *
 * using projected gradient descent.
 */

#include <Eigen/Dense>

namespace finance {

/**
 * @brief Compute minimum-variance (or mean-variance) portfolio weights.
 *
 * @param returns_matrix  (T × N) matrix of asset returns (rows=time, cols=assets).
 * @param risk_aversion   Risk-aversion coefficient γ.  Set to 0 for pure
 *                        minimum-variance.  Higher values tilt towards return.
 * @return Eigen::VectorXd  Optimal weight vector of length N.
 */
Eigen::VectorXd optimize_weights(
    const Eigen::MatrixXd& returns_matrix,
    double risk_aversion = 0.0);

} // namespace finance
