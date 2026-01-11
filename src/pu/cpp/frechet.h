#ifndef PU_FRECHET_H
#define PU_FRECHET_H

#include <Eigen/Dense>

namespace pu {

struct FrechetResult {
    double distance;
    Eigen::VectorXd mean1;
    Eigen::VectorXd mean2;
    double trace_cov1;
    double trace_cov2;
    double trace_sqrt_product;
};

/**
 * Compute Fréchet distance between two Gaussian distributions.
 *
 * FD = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2(Σ1 Σ2)^{1/2})
 *
 * Similar to FID (Fréchet Inception Distance) but for arbitrary embeddings.
 */
FrechetResult frechet_distance(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

/**
 * Compute matrix square root using eigendecomposition.
 */
Eigen::MatrixXd matrix_sqrt(const Eigen::MatrixXd& A);

} // namespace pu

#endif // PU_FRECHET_H
