/**
 * Linear R² (coefficient of determination) implementation using Eigen
 */

#include "linear_r2.h"
#include <stdexcept>
#include <cmath>

namespace pu {

// Compute R² for predicting Y from X using linear regression
double compute_r2_direction(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    // Center the matrices
    Eigen::MatrixXd X_centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd Y_centered = Y.rowwise() - Y.colwise().mean();

    // Solve least squares: find W such that Y ≈ X @ W
    // Using normal equations: W = (X^T X)^{-1} X^T Y
    Eigen::MatrixXd XtX = X_centered.transpose() * X_centered;

    // Add small regularization for numerical stability
    XtX += 1e-6 * Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());

    Eigen::MatrixXd XtY = X_centered.transpose() * Y_centered;
    Eigen::MatrixXd W = XtX.ldlt().solve(XtY);

    // Predict Y
    Eigen::MatrixXd Y_pred = X_centered * W;

    // Compute R²
    double ss_res = (Y_centered - Y_pred).squaredNorm();
    double ss_tot = Y_centered.squaredNorm();

    if (ss_tot < 1e-10) {
        return 1.0;  // Perfect prediction if Y is constant
    }

    double r2 = 1.0 - (ss_res / ss_tot);
    return std::max(0.0, r2);  // Clamp to non-negative
}

double linear_r2(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (X.rows() != Y.rows()) {
        throw std::runtime_error("Matrices must have same number of rows");
    }
    return compute_r2_direction(X, Y);
}

LinearR2Result bidirectional_linear_r2(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (X.rows() != Y.rows()) {
        throw std::runtime_error("Matrices must have same number of rows");
    }

    LinearR2Result result;
    result.r2_x_to_y = compute_r2_direction(X, Y);
    result.r2_y_to_x = compute_r2_direction(Y, X);
    result.r2 = (result.r2_x_to_y + result.r2_y_to_x) / 2.0;

    // Compute RMSE for X->Y direction
    Eigen::MatrixXd X_centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd Y_centered = Y.rowwise() - Y.colwise().mean();
    Eigen::MatrixXd XtX = X_centered.transpose() * X_centered;
    XtX += 1e-6 * Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());
    Eigen::MatrixXd W = XtX.ldlt().solve(X_centered.transpose() * Y_centered);
    Eigen::MatrixXd Y_pred = X_centered * W;
    result.rmse = std::sqrt((Y_centered - Y_pred).squaredNorm() / Y.rows());

    return result;
}

} // namespace pu
