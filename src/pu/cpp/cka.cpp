/*
 * Central Kernel Alignment (CKA) computation using memory-mapped files
 * Handles large matrices (~80GB) that don't fit in memory
 *
 * Usage: ./cka_mmap <matrix1.bin> <matrix2.bin> <n> <m>
 *   - matrix1.bin, matrix2.bin: binary files containing row-major double matrices
 *   - n: number of rows
 *   - m: number of columns
 *
 * CKA formula: CKA(K,L) = <K_c, L_c>_F / (||K_c||_F * ||L_c||_F)
 * where K_c, L_c are centered versions and <·,·>_F is Frobenius inner product
 */

#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iomanip>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cka.h"

class MMapMatrix {
private:
    int fd;
    double* data;
    size_t file_size;
    size_t n_rows;
    size_t n_cols;

public:
    MMapMatrix(const char* filename, size_t rows, size_t cols)
        : fd(-1), data(nullptr), n_rows(rows), n_cols(cols) {

        file_size = rows * cols * sizeof(double);

        // Open file
        fd = open(filename, O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error(std::string("Failed to open file: ") + filename);
        }

        // Verify file size
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Failed to stat file");
        }

        if ((size_t)sb.st_size != file_size) {
            close(fd);
            throw std::runtime_error("File size mismatch. Expected " +
                std::to_string(file_size) + " bytes, got " + std::to_string(sb.st_size));
        }

        // Memory map the file
        data = (double*)mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to mmap file");
        }

        // Advise kernel about access pattern
        madvise(data, file_size, MADV_SEQUENTIAL | MADV_WILLNEED);
    }

    ~MMapMatrix() {
        if (data != nullptr && data != MAP_FAILED) {
            munmap(data, file_size);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    // Prevent copying
    MMapMatrix(const MMapMatrix&) = delete;
    MMapMatrix& operator=(const MMapMatrix&) = delete;

    inline double operator()(size_t i, size_t j) const {
        return data[i * n_cols + j];
    }

    inline const double* row_ptr(size_t i) const {
        return data + i * n_cols;
    }

    size_t rows() const { return n_rows; }
    size_t cols() const { return n_cols; }
};

class CKAComputer {
private:
    const MMapMatrix& K;
    const MMapMatrix& L;
    size_t n;
    size_t m;

    // Statistics for centering
    std::vector<double> K_row_means;
    std::vector<double> L_row_means;
    double K_grand_mean;
    double L_grand_mean;

    // Compute row means and grand mean for a matrix
    void compute_means(const MMapMatrix& mat, std::vector<double>& row_means, double& grand_mean) {
        size_t rows = mat.rows();
        size_t cols = mat.cols();
        row_means.resize(rows);

        double sum_all = 0.0;

        #pragma omp parallel for reduction(+:sum_all) schedule(dynamic, 64)
        for (size_t i = 0; i < rows; i++) {
            double row_sum = 0.0;
            const double* row = mat.row_ptr(i);

            for (size_t j = 0; j < cols; j++) {
                row_sum += row[j];
            }

            row_means[i] = row_sum / cols;
            sum_all += row_sum;
        }

        grand_mean = sum_all / (rows * cols);
    }

    // Get centered value: X_centered[i,j] = X[i,j] - row_mean[i] - col_mean[j] + grand_mean
    // For symmetric matrices (kernels), col_mean[j] = row_mean[j]
    inline double get_centered_K(size_t i, size_t j) const {
        return K(i, j) - K_row_means[i] - K_row_means[j] + K_grand_mean;
    }

    inline double get_centered_L(size_t i, size_t j) const {
        return L(i, j) - L_row_means[i] - L_row_means[j] + L_grand_mean;
    }

public:
    CKAComputer(const MMapMatrix& K_, const MMapMatrix& L_)
        : K(K_), L(L_), n(K_.rows()), m(K_.cols()) {

        if (K.rows() != L.rows() || K.cols() != L.cols()) {
            throw std::runtime_error("Matrix dimensions must match");
        }

        std::cerr << "Computing means for matrix K..." << std::endl;
        compute_means(K, K_row_means, K_grand_mean);

        std::cerr << "Computing means for matrix L..." << std::endl;
        compute_means(L, L_row_means, L_grand_mean);
    }

    double compute_cka() {
        std::cerr << "Computing Frobenius inner products..." << std::endl;

        // We need to compute:
        // numerator = <K_c, L_c>_F = sum_ij K_c[i,j] * L_c[i,j]
        // norm_K = ||K_c||_F = sqrt(sum_ij K_c[i,j]^2)
        // norm_L = ||L_c||_F = sqrt(sum_ij L_c[i,j]^2)

        double inner_product = 0.0;
        double norm_K_sq = 0.0;
        double norm_L_sq = 0.0;

        // Process in chunks to maintain cache locality
        const size_t CHUNK_SIZE = 1024;

        #pragma omp parallel for reduction(+:inner_product,norm_K_sq,norm_L_sq) schedule(dynamic, 16)
        for (size_t i_start = 0; i_start < n; i_start += CHUNK_SIZE) {
            size_t i_end = std::min(i_start + CHUNK_SIZE, n);

            for (size_t i = i_start; i < i_end; i++) {
                // Prefetch next rows for better cache performance
                if (i + 1 < i_end) {
                    __builtin_prefetch(K.row_ptr(i + 1), 0, 1);
                    __builtin_prefetch(L.row_ptr(i + 1), 0, 1);
                }

                const double* K_row = K.row_ptr(i);
                const double* L_row = L.row_ptr(i);
                const double K_row_mean = K_row_means[i];
                const double L_row_mean = L_row_means[i];

                for (size_t j = 0; j < m; j++) {
                    // Centered values
                    double K_c = K_row[j] - K_row_mean - K_row_means[j] + K_grand_mean;
                    double L_c = L_row[j] - L_row_mean - L_row_means[j] + L_grand_mean;

                    inner_product += K_c * L_c;
                    norm_K_sq += K_c * K_c;
                    norm_L_sq += L_c * L_c;
                }
            }

            // Progress indicator for large matrices
            if (i_start % (CHUNK_SIZE * 100) == 0) {
                double progress = 100.0 * i_start / n;
                #pragma omp critical
                {
                    std::cerr << "\rProgress: " << std::fixed << std::setprecision(1)
                              << progress << "%" << std::flush;
                }
            }
        }

        std::cerr << "\rProgress: 100.0%       " << std::endl;

        double norm_K = std::sqrt(norm_K_sq);
        double norm_L = std::sqrt(norm_L_sq);

        if (norm_K < 1e-10 || norm_L < 1e-10) {
            throw std::runtime_error("One of the matrices has near-zero norm after centering");
        }

        double cka = inner_product / (norm_K * norm_L);

        // Output detailed statistics
        std::cerr << "\nStatistics:" << std::endl;
        std::cerr << "  K grand mean: " << K_grand_mean << std::endl;
        std::cerr << "  L grand mean: " << L_grand_mean << std::endl;
        std::cerr << "  ||K_c||_F: " << norm_K << std::endl;
        std::cerr << "  ||L_c||_F: " << norm_L << std::endl;
        std::cerr << "  <K_c, L_c>: " << inner_product << std::endl;

        return cka;
    }
};

double compute_cka_from_files(const std::string& file1,
                              const std::string& file2,
                              size_t n_rows,
                              size_t n_cols) {
    // Use the existing classes (MMapMatrix, CKAComputer)
    MMapMatrix K(file1.c_str(), n_rows, n_cols);
    MMapMatrix L(file2.c_str(), n_rows, n_cols);
    CKAComputer computer(K, L);
    return computer.compute_cka();
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <matrix1.bin> <matrix2.bin> <n_rows> <n_cols>\n";
        std::cerr << "\n";
        std::cerr << "Arguments:\n";
        std::cerr << "  matrix1.bin  - First matrix in binary format (row-major double)\n";
        std::cerr << "  matrix2.bin  - Second matrix in binary format (row-major double)\n";
        std::cerr << "  n_rows       - Number of rows in each matrix\n";
        std::cerr << "  n_cols       - Number of columns in each matrix\n";
        std::cerr << "\n";
        std::cerr << "Example:\n";
        std::cerr << "  " << argv[0] << " kernel1.bin kernel2.bin 100000 100000\n";
        return 1;
    }

    const char* file1 = argv[1];
    const char* file2 = argv[2];
    size_t n_rows = std::stoull(argv[3]);
    size_t n_cols = std::stoull(argv[4]);

    try {
        #ifdef _OPENMP
        std::cerr << "OpenMP enabled with " << omp_get_max_threads() << " threads" << std::endl;
        #else
        std::cerr << "OpenMP not enabled - running single-threaded" << std::endl;
        #endif

        std::cerr << "Matrix dimensions: " << n_rows << " x " << n_cols << std::endl;
        std::cerr << "Expected file size: " << (n_rows * n_cols * sizeof(double) / (1024.0 * 1024.0 * 1024.0))
                  << " GB per file" << std::endl;

        std::cerr << "\nMemory mapping matrix 1..." << std::endl;
        MMapMatrix K(file1, n_rows, n_cols);

        std::cerr << "Memory mapping matrix 2..." << std::endl;
        MMapMatrix L(file2, n_rows, n_cols);

        std::cerr << "\nComputing CKA..." << std::endl;
        CKAComputer computer(K, L);
        double cka = computer.compute_cka();

        std::cout << "\n========================================\n";
        std::cout << "CKA Score: " << std::setprecision(10) << cka << std::endl;
        std::cout << "========================================\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}