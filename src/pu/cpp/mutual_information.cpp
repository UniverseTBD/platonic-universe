/**
 * Mutual Information for measuring dependence between embedding spaces
 *
 * Assumes Gaussian distributions for closed-form computation.
 */

#include "mutual_information.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace pu {

MutualInformationResult compute_mutual_information(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double eps
) {
    if (X.rows() != Y.rows()) {
        throw std::runtime_error("Number of samples must match");
    }

    int n = static_cast<int>(X.rows());
    int d_x = static_cast<int>(X.cols());
    int d_y = static_cast<int>(Y.cols());

    // Compute marginal covariances
    Eigen::RowVectorXd mean_X = X.colwise().mean();
    Eigen::RowVectorXd mean_Y = Y.colwise().mean();

    Eigen::MatrixXd X_centered = X.rowwise() - mean_X;
    Eigen::MatrixXd Y_centered = Y.rowwise() - mean_Y;

    Eigen::MatrixXd Sigma_X = (X_centered.transpose() * X_centered) / (n - 1);
    Eigen::MatrixXd Sigma_Y = (Y_centered.transpose() * Y_centered) / (n - 1);

    Sigma_X += eps * Eigen::MatrixXd::Identity(d_x, d_x);
    Sigma_Y += eps * Eigen::MatrixXd::Identity(d_y, d_y);

    // Compute joint covariance
    Eigen::MatrixXd XY(n, d_x + d_y);
    XY << X, Y;
    Eigen::RowVectorXd mean_XY = XY.colwise().mean();
    Eigen::MatrixXd XY_centered = XY.rowwise() - mean_XY;
    Eigen::MatrixXd Sigma_XY = (XY_centered.transpose() * XY_centered) / (n - 1);
    Sigma_XY += eps * Eigen::MatrixXd::Identity(d_x + d_y, d_x + d_y);

    // Compute log determinants
    double logdet_X = std::log(Sigma_X.determinant());
    double logdet_Y = std::log(Sigma_Y.determinant());
    double logdet_XY = std::log(Sigma_XY.determinant());

    // Mutual information: MI = 0.5 * (log|Sigma_X| + log|Sigma_Y| - log|Sigma_XY|)
    double mi = 0.5 * (logdet_X + logdet_Y - logdet_XY);

    // Compute entropies (for Gaussian: H = 0.5*d*(1+log(2*pi)) + 0.5*log|Sigma|)
    double H_X = 0.5 * d_x * (1 + std::log(2 * M_PI)) + 0.5 * logdet_X;
    double H_Y = 0.5 * d_y * (1 + std::log(2 * M_PI)) + 0.5 * logdet_Y;
    double H_XY = 0.5 * (d_x + d_y) * (1 + std::log(2 * M_PI)) + 0.5 * logdet_XY;

    // Normalized MI
    double min_entropy = std::min(H_X, H_Y);
    double normalized_mi = (min_entropy > 0) ? mi / min_entropy : 0.0;

    MutualInformationResult result;
    result.mutual_information = mi;
    result.mutual_information_bits = mi / std::log(2.0);
    result.normalized_mi = normalized_mi;
    result.H_X = H_X;
    result.H_Y = H_Y;
    result.H_XY = H_XY;

    return result;
}

} // namespace pu
