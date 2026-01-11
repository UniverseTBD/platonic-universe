#ifndef PU_SVCCA_H
#define PU_SVCCA_H

#include <Eigen/Dense>
#include <vector>

namespace pu {

struct SVCCAResult {
    double similarity;
    Eigen::VectorXd correlations;
    int n_components_x;
    int n_components_y;
    double variance_explained_x;
    double variance_explained_y;
};

/**
 * Compute SVCCA (Singular Vector Canonical Correlation Analysis) between two embedding matrices.
 *
 * SVCCA = SVD preprocessing + CCA
 * 1. Performs SVD on each embedding matrix
 * 2. Keeps top components that explain variance_threshold of variance
 * 3. Runs CCA on the reduced representations
 *
 * @param X First embedding matrix (n_samples x n_features1)
 * @param Y Second embedding matrix (n_samples x n_features2)
 * @param variance_threshold Fraction of variance to retain (default: 0.99)
 * @return SVCCAResult containing similarity score and metadata
 */
SVCCAResult compute_svcca(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double variance_threshold = 0.99
);

/**
 * Compute CCA (Canonical Correlation Analysis) between two centered matrices.
 *
 * @param X First matrix (n_features1 x n_samples), already centered
 * @param Y Second matrix (n_features2 x n_samples), already centered
 * @return Vector of canonical correlations
 */
Eigen::VectorXd compute_cca(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

} // namespace pu

#endif // PU_SVCCA_H
