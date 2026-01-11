/**
 * Cosine Similarity metrics implementation using Eigen
 *
 * Computes various cosine-based similarity and distance measures
 * between embedding matrices.
 */

#include "cosine.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pu {

// Helper to compute statistics from a vector
CosineStats compute_stats(const Eigen::VectorXd& values) {
    CosineStats stats;

    stats.mean = values.mean();
    stats.min_val = values.minCoeff();
    stats.max_val = values.maxCoeff();

    // Standard deviation
    double variance = (values.array() - stats.mean).square().mean();
    stats.std_dev = std::sqrt(variance);

    // Median
    std::vector<double> sorted_values(values.data(), values.data() + values.size());
    std::sort(sorted_values.begin(), sorted_values.end());
    size_t n = sorted_values.size();
    if (n % 2 == 0) {
        stats.median = (sorted_values[n/2 - 1] + sorted_values[n/2]) / 2.0;
    } else {
        stats.median = sorted_values[n/2];
    }

    return stats;
}

Eigen::VectorXd compute_cosine_similarities(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    if (X.rows() != Y.rows()) {
        throw std::runtime_error("Matrices must have same number of rows");
    }

    Eigen::Index n_samples = X.rows();
    Eigen::Index min_features = std::min(X.cols(), Y.cols());

    // Truncate to minimum features
    Eigen::MatrixXd X_trunc = X.leftCols(min_features);
    Eigen::MatrixXd Y_trunc = Y.leftCols(min_features);

    Eigen::VectorXd cosine_sims(n_samples);

    #pragma omp parallel for
    for (Eigen::Index i = 0; i < n_samples; ++i) {
        Eigen::VectorXd x = X_trunc.row(i);
        Eigen::VectorXd y = Y_trunc.row(i);

        double norm_x = x.norm();
        double norm_y = y.norm();

        if (norm_x < 1e-10 || norm_y < 1e-10) {
            cosine_sims(i) = 0.0;
        } else {
            cosine_sims(i) = x.dot(y) / (norm_x * norm_y);
        }
    }

    return cosine_sims;
}

CosineResult mean_cosine_similarity(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    Eigen::VectorXd cosine_sims = compute_cosine_similarities(X, Y);

    CosineResult result;
    result.stats = compute_stats(cosine_sims);
    result.value = result.stats.mean;

    return result;
}

CosineResult mean_cosine_distance(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    Eigen::VectorXd cosine_sims = compute_cosine_similarities(X, Y);
    Eigen::VectorXd cosine_dists = 1.0 - cosine_sims.array();

    CosineResult result;
    result.stats = compute_stats(cosine_dists);
    result.value = result.stats.mean;

    return result;
}

double centered_cosine_similarity(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    // Center the matrices
    Eigen::MatrixXd X_centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd Y_centered = Y.rowwise() - Y.colwise().mean();

    // Compute mean cosine similarity on centered data
    CosineResult result = mean_cosine_similarity(X_centered, Y_centered);
    return result.value;
}

CosineResult mean_angular_distance(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y
) {
    Eigen::VectorXd cosine_sims = compute_cosine_similarities(X, Y);

    // Clip to valid range for arccos
    cosine_sims = cosine_sims.array().max(-1.0).min(1.0);

    // Compute angular distances (normalized to [0, 1])
    Eigen::VectorXd angular_dists = cosine_sims.array().acos() / M_PI;

    CosineResult result;
    result.stats = compute_stats(angular_dists);
    result.value = result.stats.mean;

    return result;
}

} // namespace pu


// Standalone CLI for testing
#ifdef BUILD_COSINE_CLI
#include <iostream>
#include <fstream>
#include <chrono>

Eigen::MatrixXd load_matrix_binary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

    Eigen::MatrixXd mat(rows, cols);
    file.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(double));

    return mat;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix1.bin> <matrix2.bin>" << std::endl;
        return 1;
    }

    std::string file1 = argv[1];
    std::string file2 = argv[2];

    try {
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd mat1 = load_matrix_binary(file1);
        Eigen::MatrixXd mat2 = load_matrix_binary(file2);

        std::cout << "Loaded matrices: " << mat1.rows() << " x " << mat1.cols() << std::endl;

        auto sim_result = pu::mean_cosine_similarity(mat1, mat2);
        auto dist_result = pu::mean_cosine_distance(mat1, mat2);
        double centered = pu::centered_cosine_similarity(mat1, mat2);
        auto angular_result = pu::mean_angular_distance(mat1, mat2);

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "\nResults:" << std::endl;
        std::cout << "  Mean Cosine Similarity: " << sim_result.value << std::endl;
        std::cout << "  Mean Cosine Distance: " << dist_result.value << std::endl;
        std::cout << "  Centered Cosine Similarity: " << centered << std::endl;
        std::cout << "  Mean Angular Distance: " << angular_result.value << std::endl;
        std::cout << "  Computation time: "
                  << std::chrono::duration<double>(end - start).count()
                  << "s" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
#endif
