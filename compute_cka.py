#!/usr/bin/env python3
"""
Compute CKA (Central Kernel Alignment) between two kernel matrices.
Reads binary matrices and computes CKA in Python for verification.
"""
import numpy as np
import argparse


def center_kernel(K):
    """Center a kernel matrix (row and column centering)."""
    n = K.shape[0]
    # Compute row means
    row_means = K.mean(axis=1, keepdims=True)
    # Compute column means
    col_means = K.mean(axis=0, keepdims=True)
    # Compute grand mean
    grand_mean = K.mean()

    # Center: K_c = K - row_means - col_means + grand_mean
    K_centered = K - row_means - col_means + grand_mean
    return K_centered


def compute_cka(K, L):
    """
    Compute CKA between two kernel matrices.

    CKA(K, L) = <K_c, L_c>_F / (||K_c||_F * ||L_c||_F)
    """
    # Center both matrices
    K_c = center_kernel(K)
    L_c = center_kernel(L)

    # Frobenius inner product
    inner_product = np.sum(K_c * L_c)

    # Frobenius norms
    norm_K = np.linalg.norm(K_c, 'fro')
    norm_L = np.linalg.norm(L_c, 'fro')

    # CKA score
    cka = inner_product / (norm_K * norm_L)

    print(f"Statistics:")
    print(f"  K grand mean: {K.mean()}")
    print(f"  L grand mean: {L.mean()}")
    print(f"  ||K_c||_F: {norm_K}")
    print(f"  ||L_c||_F: {norm_L}")
    print(f"  <K_c, L_c>: {inner_product}")

    return cka


def main():
    parser = argparse.ArgumentParser(
        description='Compute CKA between two binary matrix files'
    )
    parser.add_argument('matrix1', help='First matrix binary file')
    parser.add_argument('matrix2', help='Second matrix binary file')
    parser.add_argument('n_rows', type=int, help='Number of rows')
    parser.add_argument('n_cols', type=int, help='Number of columns')

    args = parser.parse_args()

    print(f"Loading matrices ({args.n_rows}x{args.n_cols})...")

    # Load binary matrices (row-major, double precision)
    K = np.fromfile(args.matrix1, dtype=np.float64).reshape(args.n_rows, args.n_cols)
    L = np.fromfile(args.matrix2, dtype=np.float64).reshape(args.n_rows, args.n_cols)

    print("Computing CKA...")
    cka_score = compute_cka(K, L)

    print(f"\n{'='*40}")
    print(f"CKA Score: {cka_score:.10f}")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
