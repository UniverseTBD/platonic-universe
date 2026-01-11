/**
 * PWCCA (Projection Weighted Canonical Correlation Analysis) implementation
 *
 * Reference: Morcos et al. "Insights on representational similarity in
 * neural networks with canonical correlation" (NeurIPS 2018)
 */

#include "pwcca.h"
#include <Eigen/QR>
#include <Eigen/SVD>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pu {

PWCCAResult compute_pwcca(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    if (X.rows() != Y.rows()) {
        throw std::runtime_error("Number of samples must match");
    }

    // Center the data (subtract column means)
    Eigen::RowVectorXd X_col_means = X.colwise().mean();
    Eigen::RowVectorXd Y_col_means = Y.colwise().mean();
    Eigen::MatrixXd X_centered = X.rowwise() - X_col_means;
    Eigen::MatrixXd Y_centered = Y.rowwise() - Y_col_means;

    // QR decomposition for numerical stability
    Eigen::HouseholderQR<Eigen::MatrixXd> qr_x(X_centered);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr_y(Y_centered);

    // Get Q matrices (thin QR)
    int rank_x = std::min(static_cast<int>(X_centered.rows()), static_cast<int>(X_centered.cols()));
    int rank_y = std::min(static_cast<int>(Y_centered.rows()), static_cast<int>(Y_centered.cols()));

    Eigen::MatrixXd Q_x = qr_x.householderQ() * Eigen::MatrixXd::Identity(X_centered.rows(), rank_x);
    Eigen::MatrixXd Q_y = qr_y.householderQ() * Eigen::MatrixXd::Identity(Y_centered.rows(), rank_y);

    // Compute cross-covariance
    Eigen::MatrixXd cross_cov = Q_x.transpose() * Q_y;

    // SVD to get canonical correlations
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(cross_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Get singular values (canonical correlations)
    Eigen::VectorXd correlations = svd.singularValues();

    // Clip to [0, 1] for numerical stability
    for (int i = 0; i < correlations.size(); ++i) {
        correlations(i) = std::min(1.0, std::max(0.0, correlations(i)));
    }

    // Get canonical directions in the QR-reduced space
    Eigen::MatrixXd U = svd.matrixU();

    // Project Q_x onto canonical directions
    Eigen::MatrixXd projections = Q_x * U;

    // Compute alpha weights (variance captured by each direction)
    Eigen::VectorXd alpha = projections.colwise().squaredNorm();
    double alpha_sum = alpha.sum();
    if (alpha_sum > 0) {
        alpha /= alpha_sum;
    }

    // PWCCA is weighted average of canonical correlations
    double pwcca_score = alpha.dot(correlations);

    // Build result
    PWCCAResult result;
    result.similarity = pwcca_score;
    result.correlations = correlations;
    result.weights = alpha;

    return result;
}

} // namespace pu
