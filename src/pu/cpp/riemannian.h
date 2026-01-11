#ifndef PU_RIEMANNIAN_H
#define PU_RIEMANNIAN_H

#include <Eigen/Dense>

namespace pu {

struct RiemannianResult {
    double affine_invariant_distance;
    double log_euclidean_distance;
    double stein_divergence;
};

/**
 * Compute Riemannian geometry-based metrics on SPD manifolds.
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (n_samples x n_features)
 * @param eps Regularization parameter
 * @return RiemannianResult with various distances
 */
RiemannianResult compute_riemannian(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double eps = 1e-10
);

double affine_invariant_distance(
    const Eigen::MatrixXd& Sigma1,
    const Eigen::MatrixXd& Sigma2,
    double eps = 1e-10
);

double log_euclidean_distance(
    const Eigen::MatrixXd& Sigma1,
    const Eigen::MatrixXd& Sigma2,
    double eps = 1e-10
);

double stein_divergence(
    const Eigen::MatrixXd& Sigma1,
    const Eigen::MatrixXd& Sigma2
);

} // namespace pu

#endif // PU_RIEMANNIAN_H
