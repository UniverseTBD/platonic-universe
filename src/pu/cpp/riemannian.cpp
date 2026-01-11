/**
 * Riemannian geometry metrics for SPD manifolds
 */

#include "riemannian.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace pu {

static Eigen::MatrixXd compute_covariance(const Eigen::MatrixXd& X, double eps) {
    int n = static_cast<int>(X.rows());
    int d = static_cast<int>(X.cols());

    Eigen::RowVectorXd mean = X.colwise().mean();
    Eigen::MatrixXd X_centered = X.rowwise() - mean;
    Eigen::MatrixXd Sigma = (X_centered.transpose() * X_centered) / (n - 1);
    Sigma += eps * Eigen::MatrixXd::Identity(d, d);

    return Sigma;
}

double affine_invariant_distance(
    const Eigen::MatrixXd& Sigma1,
    const Eigen::MatrixXd& Sigma2,
    double eps
) {
    Eigen::MatrixXd Sigma1_inv = Sigma1.inverse();
    Eigen::MatrixXd M = Sigma1_inv * Sigma2;

    Eigen::EigenSolver<Eigen::MatrixXd> solver(M);
    Eigen::VectorXcd eigenvalues = solver.eigenvalues();

    double sum_log_sq = 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        double real_eig = std::max(eigenvalues(i).real(), eps);
        double log_eig = std::log(real_eig);
        sum_log_sq += log_eig * log_eig;
    }

    return std::sqrt(sum_log_sq);
}

double log_euclidean_distance(
    const Eigen::MatrixXd& Sigma1,
    const Eigen::MatrixXd& Sigma2,
    double eps
) {
    // Compute log(Sigma1) via eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver1(Sigma1);
    Eigen::VectorXd eig1 = solver1.eigenvalues();
    Eigen::MatrixXd V1 = solver1.eigenvectors();

    Eigen::VectorXd log_eig1(eig1.size());
    for (int i = 0; i < eig1.size(); ++i) {
        log_eig1(i) = std::log(std::max(eig1(i), eps));
    }
    Eigen::MatrixXd logSigma1 = V1 * log_eig1.asDiagonal() * V1.transpose();

    // Compute log(Sigma2)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver2(Sigma2);
    Eigen::VectorXd eig2 = solver2.eigenvalues();
    Eigen::MatrixXd V2 = solver2.eigenvectors();

    Eigen::VectorXd log_eig2(eig2.size());
    for (int i = 0; i < eig2.size(); ++i) {
        log_eig2(i) = std::log(std::max(eig2(i), eps));
    }
    Eigen::MatrixXd logSigma2 = V2 * log_eig2.asDiagonal() * V2.transpose();

    return (logSigma1 - logSigma2).norm();
}

double stein_divergence(
    const Eigen::MatrixXd& Sigma1,
    const Eigen::MatrixXd& Sigma2
) {
    double logdet1 = std::log(Sigma1.determinant());
    double logdet2 = std::log(Sigma2.determinant());

    Eigen::MatrixXd Sigma_mix = 0.5 * (Sigma1 + Sigma2);
    double logdet_mix = std::log(Sigma_mix.determinant());

    return logdet_mix - 0.5 * logdet1 - 0.5 * logdet2;
}

RiemannianResult compute_riemannian(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double eps
) {
    if (X.cols() != Y.cols()) {
        throw std::runtime_error("Feature dimensions must match");
    }

    Eigen::MatrixXd Sigma1 = compute_covariance(X, eps);
    Eigen::MatrixXd Sigma2 = compute_covariance(Y, eps);

    RiemannianResult result;
    result.affine_invariant_distance = affine_invariant_distance(Sigma1, Sigma2, eps);
    result.log_euclidean_distance = log_euclidean_distance(Sigma1, Sigma2, eps);
    result.stein_divergence = stein_divergence(Sigma1, Sigma2);

    return result;
}

} // namespace pu
