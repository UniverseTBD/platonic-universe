#ifndef PU_RSA_H
#define PU_RSA_H

#include <Eigen/Dense>
#include <string>

namespace pu {

struct RSAResult {
    double correlation;
    double p_value;
    std::string method;
    std::string distance_metric;
};

/**
 * Compute Representational Dissimilarity Matrix (RDM).
 *
 * @param embeddings Embedding matrix (n_samples x n_features)
 * @param metric Distance metric: "correlation", "euclidean", "cosine"
 * @return RDM matrix (n_samples x n_samples)
 */
Eigen::MatrixXd compute_rdm(
    const Eigen::MatrixXd& embeddings,
    const std::string& metric = "correlation"
);

/**
 * Compute RSA similarity between two embedding spaces.
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (n_samples x n_features)
 * @param method Correlation method: "spearman" or "pearson"
 * @param distance_metric Distance metric for RDM: "correlation", "euclidean", "cosine"
 * @return RSAResult with correlation, p_value, and method info
 */
RSAResult rsa_similarity(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const std::string& method = "spearman",
    const std::string& distance_metric = "correlation"
);

/**
 * Compute Spearman rank correlation between two vectors.
 */
std::pair<double, double> spearman_correlation(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y
);

/**
 * Compute Pearson correlation between two vectors.
 */
std::pair<double, double> pearson_correlation(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y
);

} // namespace pu

#endif // PU_RSA_H
