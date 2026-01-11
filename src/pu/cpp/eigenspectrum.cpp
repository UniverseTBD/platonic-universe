/**
 * Eigenspectrum comparison metrics implementation
 *
 * Computes various spectral similarity measures between embedding matrices
 * based on their covariance eigenvalue spectra.
 */

#include "eigenspectrum.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pu {

Eigen::VectorXd compute_sorted_eigenvalues(
    const Eigen::MatrixXd& X,
    double eps
) {
    int n = static_cast<int>(X.rows());
    int d = static_cast<int>(X.cols());

    // Center the data
    Eigen::RowVectorXd mean = X.colwise().mean();
    Eigen::MatrixXd X_centered = X.rowwise() - mean;

    // Compute covariance matrix
    Eigen::MatrixXd Sigma = (X_centered.transpose() * X_centered) / (n - 1);

    // Add regularization
    Sigma += eps * Eigen::MatrixXd::Identity(d, d);

    // Compute eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Sigma);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();

    // Sort in descending order
    std::sort(eigenvalues.data(), eigenvalues.data() + eigenvalues.size(),
              [](double a, double b) { return a > b; });

    return eigenvalues;
}

double compute_effective_rank(
    const Eigen::VectorXd& eigenvalues,
    double eps
) {
    // Ensure positivity
    Eigen::VectorXd pos_eigenvalues(eigenvalues.size());
    for (int i = 0; i < eigenvalues.size(); ++i) {
        pos_eigenvalues(i) = std::max(eigenvalues(i), eps);
    }

    // Normalize to probability distribution
    double sum = pos_eigenvalues.sum();
    Eigen::VectorXd p = pos_eigenvalues / sum;

    // Compute entropy
    double entropy = 0.0;
    for (int i = 0; i < p.size(); ++i) {
        if (p(i) > eps) {
            entropy -= p(i) * std::log(p(i));
        }
    }

    // Effective rank is exponential of entropy
    return std::exp(entropy);
}

EigenspectrumResult compute_eigenspectrum(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double eps
) {
    if (X.cols() != Y.cols()) {
        throw std::runtime_error("Feature dimensions must match");
    }

    // Compute sorted eigenvalues
    Eigen::VectorXd lambda_X = compute_sorted_eigenvalues(X, eps);
    Eigen::VectorXd lambda_Y = compute_sorted_eigenvalues(Y, eps);

    EigenspectrumResult result;
    result.eigenvalues_x = lambda_X;
    result.eigenvalues_y = lambda_Y;

    // Spectral distance (L2 norm of eigenvalue difference)
    result.spectral_distance = (lambda_X - lambda_Y).norm();

    // Log-spectral distance (scale-invariant)
    Eigen::VectorXd log_lambda_X(lambda_X.size());
    Eigen::VectorXd log_lambda_Y(lambda_Y.size());
    for (int i = 0; i < lambda_X.size(); ++i) {
        log_lambda_X(i) = std::log(std::max(lambda_X(i), eps));
        log_lambda_Y(i) = std::log(std::max(lambda_Y(i), eps));
    }
    result.log_spectral_distance = (log_lambda_X - log_lambda_Y).norm();

    // Spectral similarity (cosine similarity of eigenvalue vectors)
    double dot_product = lambda_X.dot(lambda_Y);
    double norm_X = lambda_X.norm();
    double norm_Y = lambda_Y.norm();
    result.spectral_similarity = dot_product / (norm_X * norm_Y + eps);

    // Effective ranks
    result.effective_rank_x = compute_effective_rank(lambda_X, eps);
    result.effective_rank_y = compute_effective_rank(lambda_Y, eps);
    result.effective_rank_distance = std::abs(result.effective_rank_x - result.effective_rank_y);

    return result;
}

} // namespace pu
