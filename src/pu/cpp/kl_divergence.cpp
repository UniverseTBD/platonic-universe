/**
 * KL Divergence for comparing embedding distributions
 *
 * Assumes Gaussian distributions for closed-form computation.
 */

#include "kl_divergence.h"
#include <cmath>
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

double kl_divergence_gaussian(
    const Eigen::MatrixXd& Sigma_P,
    const Eigen::MatrixXd& Sigma_Q,
    const Eigen::VectorXd& mu_P,
    const Eigen::VectorXd& mu_Q,
    double& log_det_term,
    double& trace_term,
    double& mahalanobis_term
) {
    int d = static_cast<int>(Sigma_P.cols());

    // 1. log(|Sigma_Q| / |Sigma_P|)
    double logdet_P = std::log(Sigma_P.determinant());
    double logdet_Q = std::log(Sigma_Q.determinant());
    log_det_term = logdet_Q - logdet_P;

    // 2. Tr(Sigma_Q^{-1} Sigma_P)
    Eigen::MatrixXd Sigma_Q_inv = Sigma_Q.inverse();
    trace_term = (Sigma_Q_inv * Sigma_P).trace();

    // 3. (mu_Q - mu_P)^T Sigma_Q^{-1} (mu_Q - mu_P)
    Eigen::VectorXd mu_diff = mu_Q - mu_P;
    mahalanobis_term = mu_diff.transpose() * Sigma_Q_inv * mu_diff;

    // KL divergence: 0.5 * [log_det + trace + mahalanobis - d]
    double kl_div = 0.5 * (log_det_term + trace_term + mahalanobis_term - d);

    return kl_div;
}

KLDivergenceResult compute_kl_divergence(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double eps
) {
    if (X.cols() != Y.cols()) {
        throw std::runtime_error("Feature dimensions must match");
    }

    int d = static_cast<int>(X.cols());

    // Compute means
    Eigen::VectorXd mu_P = X.colwise().mean().transpose();
    Eigen::VectorXd mu_Q = Y.colwise().mean().transpose();

    // Compute covariances with regularization
    Eigen::MatrixXd Sigma_P = compute_covariance(X, eps);
    Eigen::MatrixXd Sigma_Q = compute_covariance(Y, eps);

    KLDivergenceResult result;

    // Compute KL(P||Q)
    result.kl_pq = kl_divergence_gaussian(
        Sigma_P, Sigma_Q, mu_P, mu_Q,
        result.log_det_term, result.trace_term, result.mahalanobis_term
    );

    // Compute KL(Q||P)
    double log_det_qp, trace_qp, mahal_qp;
    result.kl_qp = kl_divergence_gaussian(
        Sigma_Q, Sigma_P, mu_Q, mu_P,
        log_det_qp, trace_qp, mahal_qp
    );

    // Symmetric KL (Jeffreys divergence)
    result.symmetric_kl = 0.5 * (result.kl_pq + result.kl_qp);

    return result;
}

} // namespace pu
