/**
 * Procrustes Analysis implementation using Eigen
 *
 * Finds optimal orthogonal transformation to align two matrices
 * and computes the normalized residual distance.
 *
 * Algorithm:
 * 1. Center both matrices
 * 2. Compute M = Y^T @ X
 * 3. SVD: M = U @ S @ V^T
 * 4. Optimal rotation: Q = U @ V^T
 * 5. Distance: ||X - Y @ Q|| / ||X||
 */

#include "procrustes.h"
#include <Eigen/SVD>
#include <stdexcept>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pu {

ProcrustesResult compute_procrustes(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    bool scale
) {
    if (X.rows() != Y.rows() || X.cols() != Y.cols()) {
        throw std::runtime_error(
            "Matrices must have same shape for Procrustes analysis"
        );
    }

    // Center the data (subtract column means)
    Eigen::MatrixXd X_centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd Y_centered = Y.rowwise() - Y.colwise().mean();

    // Optionally scale to unit norm
    if (scale) {
        double X_norm = X_centered.norm();
        double Y_norm = Y_centered.norm();
        if (X_norm > 1e-10) X_centered /= X_norm;
        if (Y_norm > 1e-10) Y_centered /= Y_norm;
    }

    // Compute M = Y^T @ X
    Eigen::MatrixXd M = Y_centered.transpose() * X_centered;

    // SVD to find optimal rotation
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd Q = svd.matrixU() * svd.matrixV().transpose();

    // Transform Y
    Eigen::MatrixXd Y_transformed = Y_centered * Q;

    // Compute residual
    Eigen::MatrixXd residual = X_centered - Y_transformed;

    // Compute normalized distance
    double X_norm = X_centered.norm();
    double procrustes_dist = 0.0;

    if (X_norm > 1e-10) {
        procrustes_dist = residual.norm() / X_norm;
    }

    // Prepare result
    ProcrustesResult result;
    result.distance = procrustes_dist;
    result.alignment_score = 1.0 - procrustes_dist;
    result.rotation_matrix = Q;
    result.mse = residual.squaredNorm() / (residual.rows() * residual.cols());
    result.frobenius_norm_residual = residual.norm();

    return result;
}

double procrustes_distance(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    bool scale
) {
    return compute_procrustes(X, Y, scale).distance;
}

} // namespace pu


// Standalone CLI for testing
#ifdef BUILD_PROCRUSTES_CLI
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
        std::cerr << "Usage: " << argv[0] << " <matrix1.bin> <matrix2.bin> [scale]" << std::endl;
        return 1;
    }

    std::string file1 = argv[1];
    std::string file2 = argv[2];
    bool scale = (argc > 3) ? (std::atoi(argv[3]) != 0) : false;

    try {
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd mat1 = load_matrix_binary(file1);
        Eigen::MatrixXd mat2 = load_matrix_binary(file2);
        auto load_end = std::chrono::high_resolution_clock::now();

        std::cout << "Loaded matrices: " << mat1.rows() << " x " << mat1.cols() << std::endl;
        std::cout << "Load time: "
                  << std::chrono::duration<double>(load_end - start).count()
                  << "s" << std::endl;

        auto result = pu::compute_procrustes(mat1, mat2, scale);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "\nResults:" << std::endl;
        std::cout << "  Procrustes distance: " << result.distance << std::endl;
        std::cout << "  Alignment score: " << result.alignment_score << std::endl;
        std::cout << "  MSE: " << result.mse << std::endl;
        std::cout << "  Frobenius norm residual: " << result.frobenius_norm_residual << std::endl;
        std::cout << "  Computation time: "
                  << std::chrono::duration<double>(end - load_end).count()
                  << "s" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
#endif
