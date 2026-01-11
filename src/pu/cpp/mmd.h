#ifndef PU_MMD_H
#define PU_MMD_H

#include <Eigen/Dense>
#include <string>

namespace pu {

struct MMDResult {
    double mmd_squared;
    double mmd;
    std::string kernel;
    double gamma;
};

/**
 * Compute squared Maximum Mean Discrepancy (MMD²) between two samples.
 *
 * @param X First sample matrix (n x d)
 * @param Y Second sample matrix (m x d)
 * @param kernel Kernel type: "rbf", "linear", "polynomial"
 * @param gamma RBF kernel bandwidth (-1 for auto/median heuristic)
 */
MMDResult mmd_squared(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const std::string& kernel = "rbf",
    double gamma = -1.0
);

/**
 * Compute unbiased MMD² estimator.
 */
MMDResult mmd_unbiased(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const std::string& kernel = "rbf",
    double gamma = -1.0
);

/**
 * Compute RBF kernel matrix.
 */
Eigen::MatrixXd rbf_kernel(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double gamma
);

} // namespace pu

#endif // PU_MMD_H
