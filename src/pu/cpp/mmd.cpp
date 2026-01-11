/**
 * Maximum Mean Discrepancy (MMD) implementation using Eigen
 */

#include "mmd.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pu {

// Compute median of pairwise distances for bandwidth selection
double median_heuristic(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    Eigen::MatrixXd combined(X.rows() + Y.rows(), X.cols());
    combined << X, Y;

    std::vector<double> dists;
    Eigen::Index n = combined.rows();

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = i + 1; j < n; ++j) {
            double d = (combined.row(i) - combined.row(j)).norm();
            dists.push_back(d);
        }
    }

    std::sort(dists.begin(), dists.end());
    double median = dists[dists.size() / 2];

    return 1.0 / (2.0 * median * median + 1e-10);
}

Eigen::MatrixXd rbf_kernel(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, double gamma) {
    Eigen::Index n = X.rows();
    Eigen::Index m = Y.rows();
    Eigen::MatrixXd K(n, m);

    #pragma omp parallel for collapse(2)
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < m; ++j) {
            double sq_dist = (X.row(i) - Y.row(j)).squaredNorm();
            K(i, j) = std::exp(-gamma * sq_dist);
        }
    }

    return K;
}

Eigen::MatrixXd linear_kernel(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    return X * Y.transpose();
}

Eigen::MatrixXd polynomial_kernel(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
                                   int degree = 3, double coef0 = 1.0) {
    Eigen::MatrixXd K = X * Y.transpose();
    K = (K.array() + coef0).pow(degree);
    return K;
}

MMDResult mmd_squared(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const std::string& kernel,
    double gamma
) {
    Eigen::Index n = X.rows();
    Eigen::Index m = Y.rows();

    // Auto gamma using median heuristic
    double used_gamma = gamma;
    if (gamma < 0 && kernel == "rbf") {
        used_gamma = median_heuristic(X, Y);
    }

    // Compute kernel matrices
    Eigen::MatrixXd K_XX, K_YY, K_XY;

    if (kernel == "rbf") {
        K_XX = rbf_kernel(X, X, used_gamma);
        K_YY = rbf_kernel(Y, Y, used_gamma);
        K_XY = rbf_kernel(X, Y, used_gamma);
    } else if (kernel == "linear") {
        K_XX = linear_kernel(X, X);
        K_YY = linear_kernel(Y, Y);
        K_XY = linear_kernel(X, Y);
    } else if (kernel == "polynomial") {
        K_XX = polynomial_kernel(X, X);
        K_YY = polynomial_kernel(Y, Y);
        K_XY = polynomial_kernel(X, Y);
    } else {
        throw std::runtime_error("Unknown kernel: " + kernel);
    }

    // Biased MMD² estimator
    double mmd_sq = K_XX.sum() / (n * n) +
                    K_YY.sum() / (m * m) -
                    2.0 * K_XY.sum() / (n * m);

    MMDResult result;
    result.mmd_squared = mmd_sq;
    result.mmd = std::sqrt(std::max(0.0, mmd_sq));
    result.kernel = kernel;
    result.gamma = used_gamma;

    return result;
}

MMDResult mmd_unbiased(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const std::string& kernel,
    double gamma
) {
    Eigen::Index n = X.rows();
    Eigen::Index m = Y.rows();

    double used_gamma = gamma;
    if (gamma < 0 && kernel == "rbf") {
        used_gamma = median_heuristic(X, Y);
    }

    Eigen::MatrixXd K_XX, K_YY, K_XY;

    if (kernel == "rbf") {
        K_XX = rbf_kernel(X, X, used_gamma);
        K_YY = rbf_kernel(Y, Y, used_gamma);
        K_XY = rbf_kernel(X, Y, used_gamma);
    } else if (kernel == "linear") {
        K_XX = linear_kernel(X, X);
        K_YY = linear_kernel(Y, Y);
        K_XY = linear_kernel(X, Y);
    } else {
        throw std::runtime_error("Unknown kernel: " + kernel);
    }

    // Unbiased estimator: exclude diagonal
    double K_XX_sum = K_XX.sum() - K_XX.trace();
    double K_YY_sum = K_YY.sum() - K_YY.trace();

    double mmd_sq = K_XX_sum / (n * (n - 1)) +
                    K_YY_sum / (m * (m - 1)) -
                    2.0 * K_XY.sum() / (n * m);

    MMDResult result;
    result.mmd_squared = mmd_sq;
    result.mmd = std::sqrt(std::max(0.0, mmd_sq));
    result.kernel = kernel;
    result.gamma = used_gamma;

    return result;
}

} // namespace pu
