/**
 * Jensen-Shannon Divergence for comparing embedding distributions
 *
 * Assumes Gaussian distributions for closed-form computation.
 */

#include "js_divergence.h"
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

// Helper: Compute KL(P||Q) for Gaussian distributions
static double kl_gaussian(
    const Eigen::VectorXd& mu1,
    const Eigen::MatrixXd& Sigma1,
    const Eigen::VectorXd& mu2,
    const Eigen::MatrixXd& Sigma2
) {
    int d = static_cast<int>(mu1.size());

    double logdet1 = std::log(Sigma1.determinant());
    double logdet2 = std::log(Sigma2.determinant());

    Eigen::MatrixXd Sigma2_inv = Sigma2.inverse();
    Eigen::VectorXd mu_diff = mu2 - mu1;

    double kl = 0.5 * (logdet2 - logdet1 +
                      (Sigma2_inv * Sigma1).trace() +
                      mu_diff.transpose() * Sigma2_inv * mu_diff - d);

    return kl;
}

JSDivergenceResult compute_js_divergence(
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

    // Mixture distribution parameters
    Eigen::VectorXd mu_M = 0.5 * (mu_P + mu_Q);
    Eigen::MatrixXd Sigma_M = 0.5 * (Sigma_P + Sigma_Q);

    // Compute KL(P||M) and KL(Q||M)
    double kl_PM = kl_gaussian(mu_P, Sigma_P, mu_M, Sigma_M);
    double kl_QM = kl_gaussian(mu_Q, Sigma_Q, mu_M, Sigma_M);

    // JS divergence
    double js_div = 0.5 * kl_PM + 0.5 * kl_QM;

    JSDivergenceResult result;
    result.js_divergence = js_div;
    result.js_divergence_normalized = js_div / std::log(2.0);
    result.js_distance = std::sqrt(js_div);
    result.kl_pm = kl_PM;
    result.kl_qm = kl_QM;

    return result;
}

} // namespace pu
