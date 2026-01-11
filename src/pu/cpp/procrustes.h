#ifndef PU_PROCRUSTES_H
#define PU_PROCRUSTES_H

#include <Eigen/Dense>

namespace pu {

struct ProcrustesResult {
    double distance;
    double alignment_score;
    Eigen::MatrixXd rotation_matrix;
    double mse;
    double frobenius_norm_residual;
};

/**
 * Compute Procrustes analysis between two embedding matrices.
 *
 * Finds the optimal orthogonal transformation (rotation/reflection)
 * to align Y to X, then measures the residual distance.
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (n_samples x n_features)
 * @param scale If true, scale matrices to unit norm before alignment
 * @return ProcrustesResult containing distance, alignment score, rotation matrix, etc.
 */
ProcrustesResult compute_procrustes(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    bool scale = false
);

/**
 * Compute only the Procrustes distance (convenience function).
 */
double procrustes_distance(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    bool scale = false
);

} // namespace pu

#endif // PU_PROCRUSTES_H
