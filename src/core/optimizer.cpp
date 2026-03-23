/**
 * optimizer.cpp
 *
 * Markowitz QP solver using projected gradient descent.
 *
 * Problem:
 *   min  w^T Σ w  −  γ × μ^T w
 *   s.t. 1^T w = 1,  w_i >= 0
 *
 * Algorithm:
 *   1. Compute sample covariance Σ and mean μ from the returns matrix.
 *   2. Run projected gradient descent with backtracking line search.
 *   3. Project onto the probability simplex after each gradient step.
 */

#include "optimizer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace finance {

namespace {

/**
 * Project a vector onto the unit probability simplex:
 *   { w : sum(w) = 1, w_i >= 0 }
 *
 * Algorithm: sort-based O(n log n) projection.
 */
Eigen::VectorXd project_simplex(const Eigen::VectorXd& v) {
    int n = static_cast<int>(v.size());
    Eigen::VectorXd u = v;
    std::sort(u.data(), u.data() + n, std::greater<double>());

    double cumsum = 0.0;
    int rho = 0;
    for (int j = 0; j < n; ++j) {
        cumsum += u(j);
        if (u(j) - (cumsum - 1.0) / (j + 1) > 0) {
            rho = j;
        }
    }

    double theta = (u.head(rho + 1).sum() - 1.0) / (rho + 1);
    Eigen::VectorXd w = (v.array() - theta).max(0.0);
    return w;
}

} // anonymous namespace

Eigen::VectorXd optimize_weights(
    const Eigen::MatrixXd& returns_matrix,
    double risk_aversion)
{
    if (returns_matrix.rows() < 2 || returns_matrix.cols() < 1) {
        throw std::invalid_argument(
            "returns_matrix must have at least 2 rows and 1 column");
    }

    int T = static_cast<int>(returns_matrix.rows());
    int N = static_cast<int>(returns_matrix.cols());

    // Compute sample mean (annualised × 252 equivalent handled in Python)
    Eigen::VectorXd mu = returns_matrix.colwise().mean();

    // Compute sample covariance Σ (unbiased, ddof=1)
    Eigen::MatrixXd centred = returns_matrix.rowwise() - mu.transpose();
    Eigen::MatrixXd cov = (centred.transpose() * centred) / (T - 1);

    // Regularise for numerical stability
    double reg = 1e-8;
    cov += reg * Eigen::MatrixXd::Identity(N, N);

    // Objective gradient: ∇f(w) = 2Σw − γμ
    auto grad = [&](const Eigen::VectorXd& w) -> Eigen::VectorXd {
        return 2.0 * cov * w - risk_aversion * mu;
    };

    // Initialise with equal weights
    Eigen::VectorXd w = Eigen::VectorXd::Constant(N, 1.0 / N);

    // Projected gradient descent with constant step size
    double step_size = 1.0 / (2.0 * cov.norm() + 1e-12);
    const int max_iter = 2000;
    const double tol = 1e-9;

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::VectorXd g = grad(w);
        Eigen::VectorXd w_new = project_simplex(w - step_size * g);

        double delta = (w_new - w).norm();
        w = w_new;
        if (delta < tol) break;
    }

    // Final normalisation
    if (w.sum() > 0) w /= w.sum();
    return w;
}

} // namespace finance
