/**
 * Tucker Congruence Coefficient implementation
 *
 * Tucker congruence measures similarity between vectors/factors.
 * It's mathematically equivalent to cosine similarity but commonly
 * used in factor analysis and psychometrics.
 */

#include "tucker.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pu {

double tucker_congruence(
    const Eigen::VectorXd& v1,
    const Eigen::VectorXd& v2
) {
    double numerator = v1.dot(v2);
    double denominator = std::sqrt(v1.squaredNorm() * v2.squaredNorm());

    if (denominator < 1e-10) {
        return 0.0;
    }

    return numerator / denominator;
}

TuckerResult tucker_columnwise(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    if (X.rows() != Y.rows()) {
        throw std::runtime_error("Number of samples must match");
    }

    int n_features = std::min(static_cast<int>(X.cols()), static_cast<int>(Y.cols()));
    Eigen::VectorXd congruences(n_features);

    #pragma omp parallel for
    for (int i = 0; i < n_features; ++i) {
        congruences(i) = tucker_congruence(X.col(i), Y.col(i));
    }

    // Compute statistics
    TuckerResult result;
    result.congruences = congruences;
    result.mean_congruence = congruences.mean();
    result.min_congruence = congruences.minCoeff();
    result.max_congruence = congruences.maxCoeff();

    // Standard deviation
    double variance = (congruences.array() - result.mean_congruence).square().mean();
    result.std_congruence = std::sqrt(variance);

    // Median
    std::vector<double> sorted(congruences.data(), congruences.data() + congruences.size());
    std::sort(sorted.begin(), sorted.end());
    int n = static_cast<int>(sorted.size());
    if (n % 2 == 0) {
        result.median_congruence = (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {
        result.median_congruence = sorted[n/2];
    }

    return result;
}

TuckerResult tucker_rowwise(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    if (X.cols() != Y.cols()) {
        throw std::runtime_error("Number of features must match for row-wise comparison");
    }
    if (X.rows() != Y.rows()) {
        throw std::runtime_error("Number of samples must match");
    }

    int n_samples = static_cast<int>(X.rows());
    Eigen::VectorXd congruences(n_samples);

    #pragma omp parallel for
    for (int i = 0; i < n_samples; ++i) {
        congruences(i) = tucker_congruence(X.row(i).transpose(), Y.row(i).transpose());
    }

    // Compute statistics
    TuckerResult result;
    result.congruences = congruences;
    result.mean_congruence = congruences.mean();
    result.min_congruence = congruences.minCoeff();
    result.max_congruence = congruences.maxCoeff();

    // Standard deviation
    double variance = (congruences.array() - result.mean_congruence).square().mean();
    result.std_congruence = std::sqrt(variance);

    // Median
    std::vector<double> sorted(congruences.data(), congruences.data() + congruences.size());
    std::sort(sorted.begin(), sorted.end());
    int n = static_cast<int>(sorted.size());
    if (n % 2 == 0) {
        result.median_congruence = (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {
        result.median_congruence = sorted[n/2];
    }

    return result;
}

double tucker_overall(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    if (X.rows() != Y.rows() || X.cols() != Y.cols()) {
        throw std::runtime_error("Matrices must have same dimensions for overall Tucker");
    }

    // Flatten matrices and compute Tucker
    Eigen::VectorXd X_flat = Eigen::Map<const Eigen::VectorXd>(X.data(), X.size());
    Eigen::VectorXd Y_flat = Eigen::Map<const Eigen::VectorXd>(Y.data(), Y.size());

    return tucker_congruence(X_flat, Y_flat);
}

} // namespace pu
