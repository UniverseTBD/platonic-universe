#!/usr/bin/env python3
"""
Generate dummy binary matrices for CKA testing.
Creates symmetric kernel matrices in row-major double precision format.
"""
import numpy as np
import argparse


def generate_kernel_matrix(n, output_file, seed=None):
    """Generate a positive semi-definite kernel matrix."""
    if seed is not None:
        np.random.seed(seed)

    # Generate random matrix and make it symmetric positive semi-definite
    # by computing K = X @ X.T
    X = np.random.randn(n, n)
    K = X @ X.T

    # Save as binary (row-major, double precision)
    K.astype(np.float64).tofile(output_file)

    size_gb = K.nbytes / (1024**3)
    print(f"Generated {output_file}: {n}x{n} matrix ({size_gb:.3f} GB)")
    return K


def main():
    parser = argparse.ArgumentParser(
        description='Generate dummy kernel matrices for CKA testing'
    )
    parser.add_argument('-n', '--size', type=int, default=1000,
                        help='Matrix size (n x n), default: 1000')
    parser.add_argument('--matrix1', type=str, default='matrix1.bin',
                        help='Output filename for first matrix')
    parser.add_argument('--matrix2', type=str, default='matrix2.bin',
                        help='Output filename for second matrix')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--identical', action='store_true',
                        help='Generate identical matrices (CKA should be 1.0)')

    args = parser.parse_args()

    print(f"Generating {args.size}x{args.size} kernel matrices...")

    K = generate_kernel_matrix(args.size, args.matrix1, seed=args.seed)

    if args.identical:
        # Copy the first matrix
        K.astype(np.float64).tofile(args.matrix2)
        print(f"Generated {args.matrix2}: identical to {args.matrix1}")
        print("Expected CKA: 1.0")
    else:
        generate_kernel_matrix(args.size, args.matrix2, seed=args.seed + 1)

    print(f"\nTo compute CKA, run:")
    print(f"  ./cka_mmap {args.matrix1} {args.matrix2} {args.size} {args.size}")


if __name__ == '__main__':
    main()
