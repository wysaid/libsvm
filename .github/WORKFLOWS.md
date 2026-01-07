# GitHub Actions Workflows

## Tests Workflow

The `tests.yml` workflow runs comprehensive tests for LibSVM on multiple platforms.

### Trigger Events

- Push to `master` or `refactor/cmake_cpp` branches
- Pull requests to `master` or `refactor/cmake_cpp` branches
- Manual trigger via GitHub Actions UI

### Jobs

#### 1. **tests** - Cross-platform Testing
Runs on: `ubuntu-latest`, `macos-latest`, `windows-latest`

Executes:
- Unit tests (80+ tests)
- Integration tests (50+ tests)
- Memory tests (45+ tests)
- Full CTest suite (193 tests)

Build configuration:
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBSVM_BUILD_TESTS=ON \
  -DLIBSVM_BUILD_APPS=ON
```

#### 2. **memory-sanitizer** - Memory Safety Testing
Runs on: `ubuntu-latest`, `macos-latest`

Tests memory safety with AddressSanitizer:
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLIBSVM_BUILD_TESTS=ON \
  -DLIBSVM_ENABLE_ASAN=ON
```

#### 3. **opencv-comparison** - OpenCV Compatibility (Optional)
Runs on: `ubuntu-latest` only

Compares LibSVM with OpenCV's SVM implementation (continues on error if OpenCV not available).

#### 4. **test-summary** - Results Summary
Aggregates results from all test jobs and reports overall status.

### Local Testing

To run the same tests locally:

```bash
# Basic tests
mkdir build && cd build
cmake -DLIBSVM_BUILD_TESTS=ON ..
cmake --build . --parallel
ctest --output-on-failure --parallel 4

# With AddressSanitizer (Linux/macOS)
cmake -DLIBSVM_BUILD_TESTS=ON -DLIBSVM_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --parallel
./bin/memory_tests

# With OpenCV comparison
cmake -DLIBSVM_BUILD_TESTS=ON -DLIBSVM_BUILD_OPENCV_COMPARISON=ON ..
cmake --build . --parallel
./bin/opencv_comparison_tests
```

### Performance

Typical execution times:
- **tests** job: ~2-3 minutes per platform
- **memory-sanitizer** job: ~3-4 minutes per platform
- **opencv-comparison** job: ~2-3 minutes

Total workflow time: ~5-7 minutes (parallel execution)

### Viewing Results

1. Go to the repository's "Actions" tab
2. Select the "Tests" workflow
3. Click on a specific workflow run
4. Review job results and test outputs

### Troubleshooting

If tests fail:
1. Click on the failed job
2. Expand the failed test step
3. Review the detailed test output
4. Run the same test locally to reproduce

For ASan failures:
```bash
# Build with ASan locally
cmake -DLIBSVM_BUILD_TESTS=ON -DLIBSVM_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
./bin/memory_tests
```

## Other Workflows

- **cmake-ci.yml** - Main CI workflow for building and testing the library
- **wheel.yml** - Python wheel building and distribution
