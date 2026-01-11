/**
 * Fréchet Distance implementation using Eigen
 *
 * Computes the Fréchet distance between two multivariate Gaussian distributions.
 */

#include "frechet.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

namespace pu {

Eigen::MatrixXd matrix_sqrt(const Eigen::MatrixXd& A) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed");
    }

    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();

    // Clamp negative eigenvalues to zero (numerical stability)
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues(i) = std::sqrt(std::max(0.0, eigenvalues(i)));
    }

    return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

FrechetResult frechet_distance(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    // Compute means
    Eigen::VectorXd mu1 = X.colwise().mean();
    Eigen::VectorXd mu2 = Y.colwise().mean();

    // Center the data
    Eigen::MatrixXd X_centered = X.rowwise() - mu1.transpose();
    Eigen::MatrixXd Y_centered = Y.rowwise() - mu2.transpose();

    // Compute covariance matrices
    Eigen::MatrixXd cov1 = (X_centered.transpose() * X_centered) / (X.rows() - 1);
    Eigen::MatrixXd cov2 = (Y_centered.transpose() * Y_centered) / (Y.rows() - 1);

    // Mean difference squared norm
    double mean_diff_sq = (mu1 - mu2).squaredNorm();

    // Trace terms
    double trace_cov1 = cov1.trace();
    double trace_cov2 = cov2.trace();

    // Compute sqrt(cov1 @ cov2) using Newton-Schulz or eigendecomposition
    // For numerical stability, use: Tr(sqrt(A @ B)) = Tr(sqrt(sqrt(A) @ B @ sqrt(A)))
    Eigen::MatrixXd sqrt_cov1 = matrix_sqrt(cov1);
    Eigen::MatrixXd product = sqrt_cov1 * cov2 * sqrt_cov1;
    Eigen::MatrixXd sqrt_product = matrix_sqrt(product);
    double trace_sqrt_product = sqrt_product.trace();

    // Fréchet distance
    double fd = mean_diff_sq + trace_cov1 + trace_cov2 - 2.0 * trace_sqrt_product;
    fd = std::max(0.0, fd);  // Numerical stability

    FrechetResult result;
    result.distance = std::sqrt(fd);
    result.mean1 = mu1;
    result.mean2 = mu2;
    result.trace_cov1 = trace_cov1;
    result.trace_cov2 = trace_cov2;
    result.trace_sqrt_product = trace_sqrt_product;

    return result;
}

} // namespace pu
