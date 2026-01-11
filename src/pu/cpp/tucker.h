#ifndef PU_TUCKER_H
#define PU_TUCKER_H

#include <Eigen/Dense>

namespace pu {

struct TuckerResult {
    double mean_congruence;
    double median_congruence;
    double std_congruence;
    double min_congruence;
    double max_congruence;
    Eigen::VectorXd congruences;
};

/**
 * Compute Tucker congruence coefficient between two vectors.
 *
 * Tucker = (x · y) / sqrt((x · x)(y · y))
 *
 * @param v1 First vector
 * @param v2 Second vector
 * @return Tucker congruence coefficient in [-1, 1]
 */
double tucker_congruence(
    const Eigen::VectorXd& v1,
    const Eigen::VectorXd& v2
);

/**
 * Compute column-wise Tucker congruence between two matrices.
 * Computes Tucker coefficient for each pair of corresponding columns.
 *
 * @param X First matrix (n_samples x n_features)
 * @param Y Second matrix (n_samples x n_features)
 * @return TuckerResult with statistics over all column pairs
 */
TuckerResult tucker_columnwise(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

/**
 * Compute row-wise Tucker congruence between two matrices.
 * Computes Tucker coefficient for each pair of corresponding rows.
 *
 * @param X First matrix (n_samples x n_features)
 * @param Y Second matrix (n_samples x n_features)
 * @return TuckerResult with statistics over all row pairs
 */
TuckerResult tucker_rowwise(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

/**
 * Compute overall Tucker congruence treating matrices as flattened vectors.
 *
 * @param X First matrix
 * @param Y Second matrix (must have same dimensions)
 * @return Overall Tucker congruence coefficient
 */
double tucker_overall(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

} // namespace pu

#endif // PU_TUCKER_H
