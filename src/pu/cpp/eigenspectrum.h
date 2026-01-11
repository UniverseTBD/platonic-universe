#ifndef PU_EIGENSPECTRUM_H
#define PU_EIGENSPECTRUM_H

#include <Eigen/Dense>

namespace pu {

struct EigenspectrumResult {
    double spectral_distance;
    double log_spectral_distance;
    double spectral_similarity;
    double effective_rank_x;
    double effective_rank_y;
    double effective_rank_distance;
    Eigen::VectorXd eigenvalues_x;
    Eigen::VectorXd eigenvalues_y;
};

/**
 * Compute eigenspectrum-based similarity metrics between two embedding matrices.
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (n_samples x n_features)
 * @param eps Regularization parameter for numerical stability
 * @return EigenspectrumResult with various spectral metrics
 */
EigenspectrumResult compute_eigenspectrum(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double eps = 1e-10
);

/**
 * Compute sorted eigenvalues of the covariance matrix.
 */
Eigen::VectorXd compute_sorted_eigenvalues(
    const Eigen::MatrixXd& X,
    double eps = 1e-10
);

/**
 * Compute effective rank (exponential of entropy of normalized eigenvalues).
 */
double compute_effective_rank(
    const Eigen::VectorXd& eigenvalues,
    double eps = 1e-10
);

} // namespace pu

#endif // PU_EIGENSPECTRUM_H
