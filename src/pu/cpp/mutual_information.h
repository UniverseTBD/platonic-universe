#ifndef PU_MUTUAL_INFORMATION_H
#define PU_MUTUAL_INFORMATION_H

#include <Eigen/Dense>

namespace pu {

struct MutualInformationResult {
    double mutual_information;       // MI(X;Y) in nats
    double mutual_information_bits;  // MI(X;Y) in bits
    double normalized_mi;            // MI / min(H(X), H(Y))
    double H_X;                      // Entropy of X
    double H_Y;                      // Entropy of Y
    double H_XY;                     // Joint entropy
};

/**
 * Compute mutual information between two embedding matrices.
 *
 * MI(X;Y) = 0.5 * log(|Sigma_X| * |Sigma_Y| / |Sigma_XY|)
 *
 * Mutual information measures the amount of information shared between
 * the two embedding spaces. It's symmetric and non-negative.
 *
 * @param X First embedding matrix (n_samples x d_x)
 * @param Y Second embedding matrix (n_samples x d_y)
 * @param eps Regularization parameter for covariance matrices
 * @return MutualInformationResult with MI, entropies, and normalized MI
 */
MutualInformationResult compute_mutual_information(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double eps = 1e-10
);

} // namespace pu

#endif // PU_MUTUAL_INFORMATION_H
