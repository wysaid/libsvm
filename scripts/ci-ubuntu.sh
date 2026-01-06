#!/bin/bash
# Ubuntu CI Simulation Script
# This script simulates the GitHub Actions CI workflow for Ubuntu

set -e  # Exit on error
set -u  # Exit on undefined variable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "LibSVM Ubuntu CI Simulation"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✓ $1${NC}"
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Cleanup function
cleanup() {
    info "Cleaning up build directories..."
    cd "$ROOT_DIR"
    rm -rf build install test_project
}

# Trap errors
trap 'echo -e "${RED}Error occurred at line $LINENO${NC}"; exit 1' ERR

# ============================================================================
# Job 1: build-and-test (Release + Debug, default compiler)
# ============================================================================
echo ""
echo "=========================================="
echo "Job 1: Build and Test (Default Compiler)"
echo "=========================================="

cleanup

for BUILD_TYPE in Release Debug; do
    info "Testing with BUILD_TYPE=$BUILD_TYPE"
    
    # Configure
    info "Configuring CMake..."
    cmake -B "$ROOT_DIR/build" \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DLIBSVM_BUILD_APPS=ON \
        -DLIBSVM_BUILD_EXAMPLES=OFF
    
    # Build
    info "Building..."
    cmake --build "$ROOT_DIR/build" --config $BUILD_TYPE
    
    # Test svm-train
    info "Testing svm-train..."
    "$ROOT_DIR/build/bin/svm-train" \
        "$ROOT_DIR/examples/data/heart_scale" \
        "$ROOT_DIR/build/heart_scale.model"
    
    # Test svm-predict
    info "Testing svm-predict..."
    "$ROOT_DIR/build/bin/svm-predict" \
        "$ROOT_DIR/examples/data/heart_scale" \
        "$ROOT_DIR/build/heart_scale.model" \
        "$ROOT_DIR/build/heart_scale.output"
    
    # Test svm-scale
    info "Testing svm-scale..."
    "$ROOT_DIR/build/bin/svm-scale" \
        -l -1 -u 1 \
        "$ROOT_DIR/examples/data/heart_scale" \
        > "$ROOT_DIR/build/heart_scale.scaled"
    
    # Verify outputs
    info "Verifying outputs..."
    if [ ! -f "$ROOT_DIR/build/heart_scale.model" ]; then
        fail "Model file not created"
    fi
    if [ ! -f "$ROOT_DIR/build/heart_scale.output" ]; then
        fail "Prediction output not created"
    fi
    
    pass "Build and test completed for $BUILD_TYPE"
    cleanup
done

# ============================================================================
# Job 2: build-and-test (GCC-13)
# ============================================================================
echo ""
echo "=========================================="
echo "Job 2: Build and Test (GCC-13)"
echo "=========================================="

cleanup
export CC=gcc-13
export CXX=g++-13

info "Configuring CMake with GCC-13..."
cmake -B "$ROOT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBSVM_BUILD_APPS=ON \
    -DLIBSVM_BUILD_EXAMPLES=OFF

info "Building..."
cmake --build "$ROOT_DIR/build" --config Release

info "Testing svm-train..."
"$ROOT_DIR/build/bin/svm-train" \
    "$ROOT_DIR/examples/data/heart_scale" \
    "$ROOT_DIR/build/heart_scale.model"

pass "GCC-13 build completed"
unset CC CXX
cleanup

# ============================================================================
# Job 3: build-and-test (Clang-18)
# ============================================================================
echo ""
echo "=========================================="
echo "Job 3: Build and Test (Clang-18)"
echo "=========================================="

cleanup
export CC=clang-18
export CXX=clang++-18

# Check if clang-18 is available
if ! command -v clang-18 &> /dev/null; then
    echo -e "${YELLOW}⚠ Clang-18 not found, skipping...${NC}"
else
    info "Configuring CMake with Clang-18..."
    cmake -B "$ROOT_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBSVM_BUILD_APPS=ON \
        -DLIBSVM_BUILD_EXAMPLES=OFF

    info "Building..."
    cmake --build "$ROOT_DIR/build" --config Release

    info "Testing svm-train..."
    "$ROOT_DIR/build/bin/svm-train" \
        "$ROOT_DIR/examples/data/heart_scale" \
        "$ROOT_DIR/build/heart_scale.model"

    pass "Clang-18 build completed"
    cleanup
fi
unset CC CXX

# ============================================================================
# Job 4: Build with OpenMP
# ============================================================================
echo ""
echo "=========================================="
echo "Job 4: Build with OpenMP"
echo "=========================================="

cleanup

info "Configuring CMake with OpenMP..."
cmake -B "$ROOT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBSVM_ENABLE_OPENMP=ON \
    -DLIBSVM_BUILD_APPS=ON

info "Building..."
cmake --build "$ROOT_DIR/build" --config Release

info "Testing with OpenMP..."
"$ROOT_DIR/build/bin/svm-train" \
    "$ROOT_DIR/examples/data/heart_scale"

pass "OpenMP build completed"
cleanup

# ============================================================================
# Job 5: Shared Library Build
# ============================================================================
echo ""
echo "=========================================="
echo "Job 5: Shared Library Build"
echo "=========================================="

cleanup

info "Configuring CMake for shared library..."
cmake -B "$ROOT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLIBSVM_BUILD_APPS=ON

info "Building..."
cmake --build "$ROOT_DIR/build" --config Release

info "Verifying shared library..."
if [ ! -f "$ROOT_DIR/build/lib/libsvm.so" ]; then
    fail "Shared library not found"
fi

pass "Shared library build completed"
cleanup

# ============================================================================
# Job 6: Python Bindings (if Python available)
# ============================================================================
echo ""
echo "=========================================="
echo "Job 6: Python Bindings"
echo "=========================================="

cleanup

if command -v python3 &> /dev/null; then
    info "Building shared library for Python..."
    cmake -B "$ROOT_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DLIBSVM_BUILD_PYTHON=ON
    cmake --build "$ROOT_DIR/build" --config Release

    info "Copying library to Python package..."
    cp "$ROOT_DIR/build/lib/libsvm.so"* "$ROOT_DIR/bindings/python/libsvm/" 2>/dev/null || \
        cp "$ROOT_DIR/build/lib/libsvm.so" "$ROOT_DIR/bindings/python/libsvm/"

    info "Testing Python import..."
    cd "$ROOT_DIR/bindings/python"
    python3 -c "from libsvm import svmutil; print('Python bindings import successful')"

    info "Testing Python training..."
    python3 -c "
from libsvm import svmutil
import sys
sys.path.insert(0, '.')
y, x = svmutil.svm_read_problem('../../examples/data/heart_scale')
prob = svmutil.svm_problem(y[:200], x[:200])
param = svmutil.svm_parameter('-q')
model = svmutil.svm_train(prob, param)
print('Python training test successful')
"
    cd "$ROOT_DIR"
    pass "Python bindings test completed"
    cleanup
else
    echo -e "${YELLOW}⚠ Python not found, skipping Python bindings test${NC}"
fi

# ============================================================================
# Job 7: Java Bindings (if Java available)
# ============================================================================
echo ""
echo "=========================================="
echo "Job 7: Java Bindings"
echo "=========================================="

cleanup

if command -v java &> /dev/null && command -v m4 &> /dev/null; then
    info "Configuring CMake for Java bindings..."
    cmake -B "$ROOT_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBSVM_BUILD_JAVA=ON

    info "Building Java bindings..."
    cmake --build "$ROOT_DIR/build" --target java-bindings

    info "Testing Java bindings..."
    java -classpath "$ROOT_DIR/build/bindings/java/libsvm.jar" svm_train \
        "$ROOT_DIR/examples/data/heart_scale" \
        "$ROOT_DIR/build/heart_scale_java.model"

    pass "Java bindings test completed"
    cleanup
else
    echo -e "${YELLOW}⚠ Java or m4 not found, skipping Java bindings test${NC}"
fi

# ============================================================================
# Job 8: Qt Example (if Qt available)
# ============================================================================
echo ""
echo "=========================================="
echo "Job 8: Qt Example"
echo "=========================================="

cleanup

if pkg-config --exists Qt5Core 2>/dev/null; then
    info "Configuring CMake for Qt example..."
    cmake -B "$ROOT_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBSVM_BUILD_EXAMPLES=ON

    info "Building svm-toy..."
    cmake --build "$ROOT_DIR/build" --target svm-toy

    info "Verifying svm-toy binary..."
    if [ ! -f "$ROOT_DIR/build/bin/svm-toy" ]; then
        fail "svm-toy binary not found"
    fi

    pass "Qt example build completed"
    cleanup
else
    echo -e "${YELLOW}⚠ Qt5 not found, skipping Qt example test${NC}"
fi

# ============================================================================
# Job 9: Static Analysis
# ============================================================================
echo ""
echo "=========================================="
echo "Job 9: Static Analysis"
echo "=========================================="

if command -v cppcheck &> /dev/null; then
    info "Running cppcheck..."
    cppcheck --enable=warning,performance,portability \
        --suppress=missingIncludeSystem \
        --error-exitcode=0 \
        src/ apps/ examples/svm-toy/ || true

    pass "Static analysis completed"
else
    echo -e "${YELLOW}⚠ cppcheck not found, skipping static analysis${NC}"
fi

# ============================================================================
# Job 10: Installation Test
# ============================================================================
echo ""
echo "=========================================="
echo "Job 10: Installation Test"
echo "=========================================="

cleanup

info "Configuring CMake for installation..."
cmake -B "$ROOT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ROOT_DIR/install" \
    -DLIBSVM_BUILD_APPS=ON

info "Building..."
cmake --build "$ROOT_DIR/build"

info "Installing..."
cmake --install "$ROOT_DIR/build"

info "Verifying installation..."
test -f "$ROOT_DIR/install/include/svm.h" || fail "svm.h not installed"
test -f "$ROOT_DIR/install/bin/svm-train" || fail "svm-train not installed"
test -f "$ROOT_DIR/install/bin/svm-predict" || fail "svm-predict not installed"
test -f "$ROOT_DIR/install/bin/svm-scale" || fail "svm-scale not installed"

info "Testing find_package..."
mkdir -p "$ROOT_DIR/test_project"
cat > "$ROOT_DIR/test_project/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(TestLibSVM)
find_package(LibSVM REQUIRED)
add_executable(test_app test.cpp)
target_link_libraries(test_app LibSVM::svm)
EOF

cat > "$ROOT_DIR/test_project/test.cpp" << 'EOF'
#include <svm.h>
int main() { return 0; }
EOF

cmake -B "$ROOT_DIR/test_project/build" \
    -S "$ROOT_DIR/test_project" \
    -DCMAKE_PREFIX_PATH="$ROOT_DIR/install"

cmake --build "$ROOT_DIR/test_project/build"

pass "Installation test completed"

# ============================================================================
# Job 11: Code Coverage
# ============================================================================
echo ""
echo "=========================================="
echo "Job 11: Code Coverage"
echo "=========================================="

cleanup

if command -v lcov &> /dev/null; then
    info "Configuring CMake with coverage..."
    cmake -B "$ROOT_DIR/build" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="--coverage" \
        -DCMAKE_C_FLAGS="--coverage" \
        -DLIBSVM_BUILD_APPS=ON

    info "Building..."
    cmake --build "$ROOT_DIR/build"

    info "Running tests..."
    "$ROOT_DIR/build/bin/svm-train" \
        "$ROOT_DIR/examples/data/heart_scale" \
        "$ROOT_DIR/build/heart_scale.model"
    "$ROOT_DIR/build/bin/svm-predict" \
        "$ROOT_DIR/examples/data/heart_scale" \
        "$ROOT_DIR/build/heart_scale.model" \
        "$ROOT_DIR/build/heart_scale.output"

    info "Generating coverage report..."
    lcov --capture --directory "$ROOT_DIR/build" \
        --output-file coverage.info
    lcov --remove coverage.info '/usr/*' '*/test/*' \
        --output-file coverage.info --ignore-errors unused || true
    lcov --list coverage.info || true

    pass "Code coverage completed"
    cleanup
else
    echo -e "${YELLOW}⚠ lcov not found, skipping code coverage${NC}"
fi

# ============================================================================
# Final Cleanup
# ============================================================================
cleanup

echo ""
echo "=========================================="
echo -e "${GREEN}✓ All CI jobs completed successfully!${NC}"
echo "=========================================="
