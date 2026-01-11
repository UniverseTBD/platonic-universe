/**
 * SVCCA (Singular Vector Canonical Correlation Analysis) implementation
 *
 * Based on: Raghu et al. "SVCCA: Singular Vector Canonical Correlation Analysis
 * for Deep Learning Dynamics and Interpretability" (NIPS 2017)
 *
 * SVCCA = SVD preprocessing + CCA
 */

#include "svcca.h"
#include <Eigen/SVD>
#include <Eigen/QR>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pu {

Eigen::VectorXd compute_cca(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    if (X.cols() != Y.cols()) {
        throw std::runtime_error("Number of samples must match");
    }

    // Center rows (subtract row means from each row)
    Eigen::VectorXd X_row_means = X.rowwise().mean();
    Eigen::VectorXd Y_row_means = Y.rowwise().mean();
    Eigen::MatrixXd X_centered = X - X_row_means.replicate(1, X.cols());
    Eigen::MatrixXd Y_centered = Y - Y_row_means.replicate(1, Y.cols());

    // QR decomposition for numerical stability
    Eigen::HouseholderQR<Eigen::MatrixXd> qr_x(X_centered.transpose());
    Eigen::HouseholderQR<Eigen::MatrixXd> qr_y(Y_centered.transpose());

    int rank_x = std::min(static_cast<int>(X_centered.cols()), static_cast<int>(X_centered.rows()));
    int rank_y = std::min(static_cast<int>(Y_centered.cols()), static_cast<int>(Y_centered.rows()));

    Eigen::MatrixXd Q_x = qr_x.householderQ() * Eigen::MatrixXd::Identity(X_centered.cols(), rank_x);
    Eigen::MatrixXd Q_y = qr_y.householderQ() * Eigen::MatrixXd::Identity(Y_centered.cols(), rank_y);

    // Cross-covariance
    Eigen::MatrixXd cross_cov = Q_x.transpose() * Q_y;

    // SVD to get canonical correlations
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(cross_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd correlations = svd.singularValues();

    // Clip to [0, 1] for numerical stability
    for (int i = 0; i < correlations.size(); ++i) {
        correlations(i) = std::min(1.0, std::max(0.0, correlations(i)));
    }

    return correlations;
}

SVCCAResult compute_svcca(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double variance_threshold
) {
    if (X.rows() != Y.rows()) {
        throw std::runtime_error("Number of samples must match");
    }

    // Transpose to features x samples (Google convention)
    Eigen::MatrixXd Xt = X.transpose();
    Eigen::MatrixXd Yt = Y.transpose();

    // Center the data (subtract row means from each row)
    Eigen::VectorXd Xt_row_means = Xt.rowwise().mean();
    Eigen::VectorXd Yt_row_means = Yt.rowwise().mean();
    Eigen::MatrixXd X_centered = Xt - Xt_row_means.replicate(1, Xt.cols());
    Eigen::MatrixXd Y_centered = Yt - Yt_row_means.replicate(1, Yt.cols());

    // SVD for dimensionality reduction
    Eigen::JacobiSVD<Eigen::MatrixXd> svd1(X_centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(Y_centered, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Get singular values and compute variance
    Eigen::VectorXd s1 = svd1.singularValues();
    Eigen::VectorXd s2 = svd2.singularValues();

    Eigen::VectorXd variance1 = s1.array().square();
    Eigen::VectorXd variance2 = s2.array().square();

    double total_var1 = variance1.sum();
    double total_var2 = variance2.sum();

    // Find number of components to keep based on variance threshold
    int n_comp1 = 0, n_comp2 = 0;
    double cumsum1 = 0.0, cumsum2 = 0.0;

    for (int i = 0; i < variance1.size(); ++i) {
        cumsum1 += variance1(i);
        if (cumsum1 / total_var1 >= variance_threshold) {
            n_comp1 = i + 1;
            break;
        }
    }
    if (n_comp1 == 0) n_comp1 = static_cast<int>(variance1.size());

    for (int i = 0; i < variance2.size(); ++i) {
        cumsum2 += variance2(i);
        if (cumsum2 / total_var2 >= variance_threshold) {
            n_comp2 = i + 1;
            break;
        }
    }
    if (n_comp2 == 0) n_comp2 = static_cast<int>(variance2.size());

    // Compute actual variance explained
    double var_explained1 = 0.0, var_explained2 = 0.0;
    for (int i = 0; i < n_comp1; ++i) {
        var_explained1 += variance1(i);
    }
    var_explained1 /= total_var1;

    for (int i = 0; i < n_comp2; ++i) {
        var_explained2 += variance2(i);
    }
    var_explained2 /= total_var2;

    // Create truncated SVD representations: S @ V^T
    // svacts = diag(s[:n_comp]) @ V[:n_comp, :].T
    Eigen::MatrixXd svacts1 = s1.head(n_comp1).asDiagonal() * svd1.matrixV().leftCols(n_comp1).transpose();
    Eigen::MatrixXd svacts2 = s2.head(n_comp2).asDiagonal() * svd2.matrixV().leftCols(n_comp2).transpose();

    // CCA on truncated representations
    Eigen::VectorXd correlations = compute_cca(svacts1, svacts2);

    // Build result
    SVCCAResult result;
    result.similarity = correlations.mean();
    result.correlations = correlations;
    result.n_components_x = n_comp1;
    result.n_components_y = n_comp2;
    result.variance_explained_x = var_explained1;
    result.variance_explained_y = var_explained2;

    return result;
}

} // namespace pu
