#ifndef PU_JS_DIVERGENCE_H
#define PU_JS_DIVERGENCE_H

#include <Eigen/Dense>

namespace pu {

struct JSDivergenceResult {
    double js_divergence;           // JS(P||Q)
    double js_divergence_normalized; // JS / log(2), in [0, 1]
    double js_distance;             // sqrt(JS), a proper metric
    double kl_pm;                   // KL(P||M)
    double kl_qm;                   // KL(Q||M)
};

/**
 * Compute Jensen-Shannon divergence between two embedding matrices.
 *
 * JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
 * where M = 0.5 * (P + Q) is the mixture distribution.
 *
 * JS divergence is symmetric and bounded in [0, log(2)].
 * sqrt(JS) is a proper metric satisfying the triangle inequality.
 *
 * @param X First embedding matrix (n_samples x n_features)
 * @param Y Second embedding matrix (m_samples x n_features)
 * @param eps Regularization parameter for covariance matrices
 * @return JSDivergenceResult with divergence, normalized values, and distance
 */
JSDivergenceResult compute_js_divergence(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    double eps = 1e-10
);

} // namespace pu

#endif // PU_JS_DIVERGENCE_H
