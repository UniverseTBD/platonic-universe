#ifndef PU_LINEAR_R2_H
#define PU_LINEAR_R2_H

#include <Eigen/Dense>

namespace pu {

struct LinearR2Result {
    double r2;
    double r2_x_to_y;
    double r2_y_to_x;
    double rmse;
};

/**
 * Compute linear R² (coefficient of determination) from X to Y.
 * Measures how well X can linearly predict Y.
 */
double linear_r2(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

/**
 * Compute bidirectional linear R² (average of both directions).
 */
LinearR2Result bidirectional_linear_r2(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
);

} // namespace pu

#endif // PU_LINEAR_R2_H
