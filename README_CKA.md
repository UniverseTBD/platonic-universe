# CKA with Memory-Mapped Files

High-performance C++ implementation of Central Kernel Alignment (CKA) for large matrices (~80GB) using mmap.

## Features

- **Memory-efficient**: Uses mmap2 to handle matrices that don't fit in RAM
- **Fast**: OpenMP parallelization + cache-optimized chunked processing
- **Numerically stable**: Careful centering and accumulation
- **Zero-copy**: Direct file access without loading into memory

## Building

```bash
make
```

This will compile with `-O3 -march=native` and OpenMP support.

## Usage

```bash
./cka_mmap <matrix1.bin> <matrix2.bin> <n_rows> <n_cols>
```

### Input Format

- Matrices must be in **binary format** (raw double precision, 8 bytes per element)
- **Row-major** layout (C-style)
- Both matrices must have the same dimensions

### Creating Binary Matrices

**From NumPy (Python):**
```python
import numpy as np

# Your matrix (e.g., n x n kernel matrix)
K = ...  # shape: (n, n)

# Save as binary
K.astype(np.float64).tofile('kernel.bin')
```

**From MATLAB:**
```matlab
% Your matrix
K = ...;  % size: [n, n]

% Save as binary
fid = fopen('kernel.bin', 'wb');
fwrite(fid, K, 'double');
fclose(fid);
```

## Example

For 100k x 100k matrices (~80GB each):

```bash
./cka_mmap kernel1.bin kernel2.bin 100000 100000
```

## Algorithm

CKA measures similarity between two kernel matrices K and L:

```
CKA(K, L) = <K_c, L_c>_F / (||K_c||_F * ||L_c||_F)
```

Where:
- `K_c`, `L_c` are row-and-column centered matrices
- `<·,·>_F` is the Frobenius inner product
- `||·||_F` is the Frobenius norm

### Implementation Details

1. **Memory mapping**: Files are mmap'd with `MAP_SHARED` and `MADV_SEQUENTIAL`
2. **Centering**: Row means computed in first pass, centering done on-the-fly
3. **Computation**: Single pass through both matrices computing all three quantities
4. **Chunking**: 1024-row chunks for cache locality
5. **Parallelization**: OpenMP parallel for with reduction

## Performance Tips

1. **SSD recommended**: For 80GB files, use fast storage
2. **Page cache**: First run may be slow as OS caches pages
3. **Threads**: Set `OMP_NUM_THREADS` to your CPU core count
4. **Huge pages**: For better TLB performance:
   ```bash
   echo 50000 | sudo tee /proc/sys/vm/nr_hugepages
   ```

## Testing

Generate small test matrices and run:
```bash
make test
```

This creates 1000x1000 test matrices and computes their CKA.

## Memory Requirements

- **RAM needed**: ~O(n) for row means + minimal overhead
- **Disk I/O**: Approximately 3 sequential passes through both files
- **For 100k x 100k matrices**:
  - Each file: 80 GB
  - RAM usage: ~1.6 MB (just the row means)
  - Time: ~10-30 minutes on modern SSD with 16 cores

## Troubleshooting

**"Failed to mmap file"**
- Check `ulimit -v` (virtual memory limit)
- Increase with `ulimit -v unlimited`

**Slow performance**
- Verify OpenMP is enabled: look for "OpenMP enabled" in output
- Check SSD vs HDD performance
- Monitor with `iostat -x 1` during execution

**Out of disk space**
- Each matrix needs n×n×8 bytes
- Ensure sufficient free space

## Technical Notes

- Uses double precision (64-bit float)
- Assumes kernel matrices (symmetric, but doesn't require it)
- Centering: row and column means subtracted, grand mean added back
- Thread-safe: reduction variables for parallel accumulation
