#ifndef PU_COSINE_H
#define PU_COSINE_H

#include <Eigen/Dense>

namespace pu {

struct CosineStats {
    double mean;
    double median;
    double std_dev;
    double min_val;
    double max_val;
};

struct CosineResult {
    double value;
    CosineStats stats;
};

/**
 * Compute mean cosine similarity between corresponding rows of two matrices.
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (n_samples x n_features)
 * @return CosineResult with mean similarity and statistics
 */
CosineResult mean_cosine_similarity(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

/**
 * Compute mean cosine distance (1 - similarity) between corresponding rows.
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (n_samples x n_features)
 * @return CosineResult with mean distance and statistics
 */
CosineResult mean_cosine_distance(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

/**
 * Compute centered cosine similarity (center matrices before computing).
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (n_samples x n_features)
 * @return Centered cosine similarity value
 */
double centered_cosine_similarity(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

/**
 * Compute mean angular distance (arccos of cosine similarity / pi).
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (n_samples x n_features)
 * @return CosineResult with mean angular distance and statistics
 */
CosineResult mean_angular_distance(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

/**
 * Compute row-wise cosine similarities (returns vector of similarities).
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (n_samples x n_features)
 * @return Vector of cosine similarities for each row pair
 */
Eigen::VectorXd compute_cosine_similarities(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

} // namespace pu

#endif // PU_COSINE_H
