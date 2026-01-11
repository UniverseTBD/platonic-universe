#ifndef PU_PWCCA_H
#define PU_PWCCA_H

#include <Eigen/Dense>

namespace pu {

struct PWCCAResult {
    double similarity;
    Eigen::VectorXd correlations;
    Eigen::VectorXd weights;
};

/**
 * Compute PWCCA (Projection Weighted Canonical Correlation Analysis).
 *
 * PWCCA combines CCA with projection-based weighting that emphasizes
 * the importance of each canonical direction based on the variance
 * in the original space that it captures.
 *
 * @param X First embedding matrix (n_samples x n_features1)
 * @param Y Second embedding matrix (n_samples x n_features2)
 * @return PWCCAResult containing similarity score, correlations, and weights
 *
 * Reference: Morcos et al. "Insights on representational similarity in
 * neural networks with canonical correlation" (NeurIPS 2018)
 */
PWCCAResult compute_pwcca(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

} // namespace pu

#endif // PU_PWCCA_H
