# LibSVM Test Suite

This directory contains comprehensive unit tests, integration tests, memory tests, and comparison tests for the LibSVM library.

## Prerequisites

- CMake 3.16+
- C++17 compatible compiler
- GoogleTest (automatically downloaded via FetchContent)

## Building Tests

From the project root directory:

```bash
mkdir build && cd build
cmake -DLIBSVM_BUILD_TESTS=ON ..
cmake --build . -j$(nproc)
```

## Running Tests

### Run all tests
```bash
cd build
ctest --output-on-failure
```

### Run specific test suites
```bash
# Unit tests only
./bin/unit_tests

# Integration tests only
./bin/integration_tests

# Memory tests only
./bin/memory_tests
```

### Filter tests by name
```bash
# Run all kernel tests
./bin/unit_tests --gtest_filter="*Kernel*"

# Run all SVM node tests
./bin/unit_tests --gtest_filter="SvmNodeTest.*"
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Testing individual components in isolation:

- **test_svm_node.cpp** - Tests for `svm_node` structure
  - Basic structure and memory layout
  - Sparse vector representation
  - Edge cases (extreme values, negative indices)

- **test_svm_parameter.cpp** - Tests for `svm_parameter` structure
  - All SVM types (C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR)
  - All kernel types (LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED)
  - Parameter validation

- **test_svm_problem.cpp** - Tests for `svm_problem` structure
  - Problem building with SvmProblemBuilder
  - Data generation utilities
  - Loading from files

- **test_kernel.cpp** - Tests for kernel functions
  - Linear, RBF, Polynomial, Sigmoid kernels
  - Kernel properties (symmetry, positive definiteness)
  - Sparse vector handling

- **test_model.cpp** - Tests for `svm_model` structure
  - Model training and properties
  - Support vector queries
  - Model memory management

### 2. Integration Tests (`tests/integration/`)

Testing complete workflows:

- **test_train_predict.cpp** - Training and prediction workflow
  - Binary/multi-class classification
  - Regression (SVR)
  - One-class SVM

- **test_cross_validation.cpp** - Cross-validation tests
  - Various k-fold values
  - Stability and reproducibility

- **test_model_io.cpp** - Model save/load operations
  - Save and reload models
  - Format compatibility
  - Error handling

- **test_probability.cpp** - Probability estimation tests
  - Probability predictions
  - Probability calibration

### 3. Memory Tests (`tests/memory/`)

Testing memory management:

- **test_memory_leaks.cpp** - Memory leak detection
  - Repeated allocation/deallocation cycles
  - ASan-compatible tests

- **test_resource_management.cpp** - RAII and resource safety
  - Exception safety
  - Concurrent access
  - Cleanup order

### 4. Comparison Tests (`tests/comparison/`)

Testing compatibility with other implementations:

- **test_upstream_comparison.cpp** - Comparison with upstream libsvm
- **test_opencv_comparison.cpp** - Comparison with OpenCV's SVM

## Special Build Options

### AddressSanitizer for Memory Tests

```bash
cmake -DLIBSVM_BUILD_TESTS=ON -DLIBSVM_ENABLE_ASAN=ON ..
cmake --build .
./bin/memory_tests
```

### Upstream Comparison Tests

To enable upstream comparison tests:

1. Add upstream libsvm to tests directory:
   ```bash
   git checkout upstream -- svm.cpp svm.h
   mkdir -p tests/upstream_libsvm
   cp svm.cpp svm.h tests/upstream_libsvm/
   git checkout master -- svm.cpp svm.h
   ```

2. Build with upstream comparison enabled:
   ```bash
   cmake -DLIBSVM_BUILD_TESTS=ON -DLIBSVM_BUILD_UPSTREAM_COMPARISON=ON ..
   cmake --build .
   ./bin/upstream_comparison_tests
   ```

### OpenCV Comparison Tests

To enable OpenCV comparison tests:

1. Install OpenCV with ml module:
   ```bash
   # Ubuntu/Debian
   sudo apt install libopencv-dev
   
   # Or build from source
   ```

2. Build with OpenCV comparison enabled:
   ```bash
   cmake -DLIBSVM_BUILD_TESTS=ON -DLIBSVM_BUILD_OPENCV_COMPARISON=ON ..
   cmake --build .
   ./bin/opencv_comparison_tests
   ```

## Test Utilities

The `tests/common/` directory contains shared test utilities:

- **SvmModelGuard** - RAII wrapper for `svm_model*`
- **SvmProblemBuilder** - Builder pattern for creating `svm_problem`
- **Data generators** - Create linearly separable, XOR, multi-class, and regression datasets
- **Metrics** - Calculate accuracy, MSE, and other metrics
- **File utilities** - Temporary file management for testing

## Test Coverage

Current test coverage includes:

| Component | Tests | Status |
|-----------|-------|--------|
| svm_node | 13 | ✅ |
| svm_parameter | 20+ | ✅ |
| svm_problem | 15+ | ✅ |
| Kernels | 20+ | ✅ |
| Model | 20+ | ✅ |
| Train/Predict | 15+ | ✅ |
| Cross-validation | 10+ | ✅ |
| Model I/O | 15+ | ✅ |
| Probability | 10+ | ✅ |
| Memory | 25+ | ✅ |
| Resource Management | 20+ | ✅ |

## Contributing

When adding new tests:

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use the `SvmProblemBuilder` for creating test data
4. Use `SvmModelGuard` for RAII-safe model management
5. Add new test files to `tests/CMakeLists.txt`

## License

See the main project LICENSE file.
