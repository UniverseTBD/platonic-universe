CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -Wall -Wextra
LDFLAGS = -fopenmp

# Detect if OpenMP is available
OPENMP_TEST := $(shell echo | $(CXX) -fopenmp -E - 2>&1 | grep -q "error" && echo "no" || echo "yes")

ifeq ($(OPENMP_TEST),yes)
    CXXFLAGS += -fopenmp
else
    $(warning OpenMP not available, compiling without parallel support)
    LDFLAGS =
endif

TARGET = cka_mmap
SOURCE = cka_mmap.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	@echo "Build complete: ./$(TARGET)"
	@echo "Usage: ./$(TARGET) <matrix1.bin> <matrix2.bin> <n_rows> <n_cols>"

clean:
	rm -f $(TARGET)

test_data:
	@echo "Generating small test matrices (1000x1000)..."
	@python3 -c "import numpy as np; \
	             np.random.seed(42); \
	             K = np.random.randn(1000, 1000); \
	             K = K @ K.T; \
	             L = np.random.randn(1000, 1000); \
	             L = L @ L.T; \
	             K.astype(np.float64).tofile('test_K.bin'); \
	             L.astype(np.float64).tofile('test_L.bin'); \
	             print('Created test_K.bin and test_L.bin')"

test: $(TARGET) test_data
	@echo "Running test..."
	./$(TARGET) test_K.bin test_L.bin 1000 1000

.PHONY: all clean test_data test
