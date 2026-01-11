/**
 * RSA (Representational Similarity Analysis) implementation using Eigen
 *
 * Computes similarity between embedding spaces by comparing their
 * Representational Dissimilarity Matrices (RDMs).
 */

#include "rsa.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pu {

// Compute pairwise Euclidean distances
Eigen::MatrixXd pairwise_euclidean(const Eigen::MatrixXd& X) {
    Eigen::Index n = X.rows();
    Eigen::MatrixXd dist(n, n);

    #pragma omp parallel for collapse(2)
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            if (i <= j) {
                double d = (X.row(i) - X.row(j)).norm();
                dist(i, j) = d;
                dist(j, i) = d;
            }
        }
    }

    return dist;
}

// Compute pairwise cosine distances
Eigen::MatrixXd pairwise_cosine(const Eigen::MatrixXd& X) {
    Eigen::Index n = X.rows();
    Eigen::MatrixXd dist(n, n);

    // Normalize rows
    Eigen::MatrixXd X_norm(n, X.cols());
    for (Eigen::Index i = 0; i < n; ++i) {
        double norm = X.row(i).norm();
        if (norm > 1e-10) {
            X_norm.row(i) = X.row(i) / norm;
        } else {
            X_norm.row(i) = X.row(i);
        }
    }

    // Compute cosine similarity matrix
    Eigen::MatrixXd sim = X_norm * X_norm.transpose();

    // Convert to distance
    dist = 1.0 - sim.array();

    return dist;
}

// Compute pairwise correlation distances
Eigen::MatrixXd pairwise_correlation(const Eigen::MatrixXd& X) {
    Eigen::Index n = X.rows();
    Eigen::MatrixXd dist(n, n);

    // Center each row (subtract row mean)
    Eigen::MatrixXd X_centered = X.rowwise() - X.colwise().mean();

    // Normalize centered rows
    Eigen::MatrixXd X_norm(n, X.cols());
    for (Eigen::Index i = 0; i < n; ++i) {
        double norm = X_centered.row(i).norm();
        if (norm > 1e-10) {
            X_norm.row(i) = X_centered.row(i) / norm;
        } else {
            X_norm.row(i) = X_centered.row(i);
        }
    }

    // Compute correlation matrix (dot product of normalized centered rows)
    Eigen::MatrixXd corr = X_norm * X_norm.transpose();

    // Convert to distance
    dist = 1.0 - corr.array();

    return dist;
}

Eigen::MatrixXd compute_rdm(
    const Eigen::MatrixXd& embeddings,
    const std::string& metric
) {
    if (metric == "euclidean") {
        return pairwise_euclidean(embeddings);
    } else if (metric == "cosine") {
        return pairwise_cosine(embeddings);
    } else if (metric == "correlation") {
        return pairwise_correlation(embeddings);
    } else {
        throw std::runtime_error("Unknown distance metric: " + metric);
    }
}

// Helper to compute ranks
Eigen::VectorXd compute_ranks(const Eigen::VectorXd& x) {
    Eigen::Index n = x.size();
    std::vector<std::pair<double, Eigen::Index>> indexed(n);

    for (Eigen::Index i = 0; i < n; ++i) {
        indexed[i] = {x(i), i};
    }

    std::sort(indexed.begin(), indexed.end());

    Eigen::VectorXd ranks(n);

    // Handle ties by averaging ranks
    Eigen::Index i = 0;
    while (i < n) {
        Eigen::Index j = i;
        while (j < n && indexed[j].first == indexed[i].first) {
            ++j;
        }
        double avg_rank = (i + j - 1) / 2.0 + 1.0;  // 1-based ranks
        for (Eigen::Index k = i; k < j; ++k) {
            ranks(indexed[k].second) = avg_rank;
        }
        i = j;
    }

    return ranks;
}

std::pair<double, double> pearson_correlation(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y
) {
    if (x.size() != y.size()) {
        throw std::runtime_error("Vectors must have same length");
    }

    Eigen::Index n = x.size();

    double mean_x = x.mean();
    double mean_y = y.mean();

    Eigen::VectorXd x_centered = x.array() - mean_x;
    Eigen::VectorXd y_centered = y.array() - mean_y;

    double numerator = x_centered.dot(y_centered);
    double denom_x = x_centered.squaredNorm();
    double denom_y = y_centered.squaredNorm();

    double r = 0.0;
    if (denom_x > 1e-10 && denom_y > 1e-10) {
        r = numerator / std::sqrt(denom_x * denom_y);
    }

    // Approximate p-value using t-distribution approximation
    double t = r * std::sqrt((n - 2) / (1 - r * r + 1e-10));
    // Simple approximation for large n
    double p_value = 2.0 * std::exp(-0.5 * t * t / (n - 2 + 1e-10));
    p_value = std::min(1.0, std::max(0.0, p_value));

    return {r, p_value};
}

std::pair<double, double> spearman_correlation(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y
) {
    Eigen::VectorXd ranks_x = compute_ranks(x);
    Eigen::VectorXd ranks_y = compute_ranks(y);

    return pearson_correlation(ranks_x, ranks_y);
}

RSAResult rsa_similarity(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const std::string& method,
    const std::string& distance_metric
) {
    if (X.rows() != Y.rows()) {
        throw std::runtime_error("Matrices must have same number of samples");
    }

    // Compute RDMs
    Eigen::MatrixXd rdm1 = compute_rdm(X, distance_metric);
    Eigen::MatrixXd rdm2 = compute_rdm(Y, distance_metric);

    Eigen::Index n = rdm1.rows();

    // Extract upper triangular values (excluding diagonal)
    Eigen::Index n_pairs = n * (n - 1) / 2;
    Eigen::VectorXd rdm1_vec(n_pairs);
    Eigen::VectorXd rdm2_vec(n_pairs);

    Eigen::Index idx = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = i + 1; j < n; ++j) {
            rdm1_vec(idx) = rdm1(i, j);
            rdm2_vec(idx) = rdm2(i, j);
            ++idx;
        }
    }

    // Compute correlation
    std::pair<double, double> corr_result;
    if (method == "spearman") {
        corr_result = spearman_correlation(rdm1_vec, rdm2_vec);
    } else if (method == "pearson") {
        corr_result = pearson_correlation(rdm1_vec, rdm2_vec);
    } else {
        throw std::runtime_error("Unknown correlation method: " + method);
    }

    RSAResult result;
    result.correlation = corr_result.first;
    result.p_value = corr_result.second;
    result.method = method;
    result.distance_metric = distance_metric;

    return result;
}

} // namespace pu


// Standalone CLI for testing
#ifdef BUILD_RSA_CLI
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
        std::cerr << "Usage: " << argv[0] << " <matrix1.bin> <matrix2.bin> [method] [metric]" << std::endl;
        return 1;
    }

    std::string file1 = argv[1];
    std::string file2 = argv[2];
    std::string method = (argc > 3) ? argv[3] : "spearman";
    std::string metric = (argc > 4) ? argv[4] : "correlation";

    try {
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd mat1 = load_matrix_binary(file1);
        Eigen::MatrixXd mat2 = load_matrix_binary(file2);

        std::cout << "Loaded matrices: " << mat1.rows() << " x " << mat1.cols() << std::endl;

        auto result = pu::rsa_similarity(mat1, mat2, method, metric);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "\nResults:" << std::endl;
        std::cout << "  RSA Correlation (" << method << "): " << result.correlation << std::endl;
        std::cout << "  P-value: " << result.p_value << std::endl;
        std::cout << "  Distance metric: " << result.distance_metric << std::endl;
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
