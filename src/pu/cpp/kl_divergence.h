#ifndef PU_KL_DIVERGENCE_H
#define PU_KL_DIVERGENCE_H

#include <Eigen/Dense>

namespace pu {

struct KLDivergenceResult {
    double kl_pq;           // KL(P||Q)
    double kl_qp;           // KL(Q||P)
    double symmetric_kl;    // 0.5 * (KL(P||Q) + KL(Q||P))
    double log_det_term;    // log(|Sigma_Q| / |Sigma_P|) for KL(P||Q)
    double trace_term;      // Tr(Sigma_Q^{-1} Sigma_P) for KL(P||Q)
    double mahalanobis_term; // (mu_Q - mu_P)^T Sigma_Q^{-1} (mu_Q - mu_P) for KL(P||Q)
};

/**
 * Compute KL divergence between two embedding matrices assuming Gaussian distributions.
 *
 * KL(P||Q) = 0.5 * [log(|Sigma_Q|/|Sigma_P|) + Tr(Sigma_Q^{-1} Sigma_P) +
 *                   (mu_Q - mu_P)^T Sigma_Q^{-1} (mu_Q - mu_P) - d]
 *
 * @param X First embedding matrix (n_samples x n_features), samples from P
 * @param Y Second embedding matrix (m_samples x n_features), samples from Q
 * @param eps Regularization parameter for covariance matrices
 * @return KLDivergenceResult with KL(P||Q), KL(Q||P), and symmetric KL
 */
KLDivergenceResult compute_kl_divergence(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double eps = 1e-10
);

/**
 * Compute one-directional KL divergence KL(P||Q).
 */
double kl_divergence_gaussian(
    const Eigen::MatrixXd& Sigma_P,
    const Eigen::MatrixXd& Sigma_Q,
    const Eigen::VectorXd& mu_P,
    const Eigen::VectorXd& mu_Q,
    double& log_det_term,
    double& trace_term,
    double& mahalanobis_term
);

} // namespace pu

#endif // PU_KL_DIVERGENCE_H
