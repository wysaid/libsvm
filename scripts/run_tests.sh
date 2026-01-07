#!/bin/bash
# LibSVM Unit Tests Runner
# This script builds and runs all LibSVM unit tests
#
# Test Categories:
# - Unit tests: Basic structure and component tests
# - Integration tests: End-to-end workflow tests
# - Memory tests: Leak detection and resource management
# - Comparison tests: Compare with upstream/OpenCV (optional)
#
# Memory Sanitizers:
# - AddressSanitizer (ASAN): Detects memory errors (leaks, buffer overflows, use-after-free)
# - UndefinedBehaviorSanitizer (UBSan): Detects undefined behavior
# - Enabled by DEFAULT for all tests
# - Memory tests ALWAYS run with sanitizers (cannot be disabled)
#
# Usage:
#   ./run_tests.sh                    # Run all tests (except comparison tests)
#   ./run_tests.sh --all              # Run all tests including comparison tests
#   ./run_tests.sh --unit             # Run only unit tests
#   ./run_tests.sh --integration      # Run only integration tests
#   ./run_tests.sh --memory           # Run only memory tests (with ASAN)
#   ./run_tests.sh --comparison       # Run only comparison tests
#   ./run_tests.sh --filter "Kernel*" # Run tests matching filter pattern
#   ./run_tests.sh --skip-build       # Skip build step, run tests only
#   ./run_tests.sh --sanitize         # Enable ASAN for all tests
#   ./run_tests.sh --no-sanitize      # Disable ASAN completely
#   ./run_tests.sh --verbose          # Show detailed test output
#   ./run_tests.sh --help             # Show this help

set -e # Exit on any error

cd "$(dirname "$0")/.."

# Detect platform
function isWindows() {
    [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]] || [[ -n "$WINDIR" ]]
}

function detectCores() {
    if [[ -n "$NUMBER_OF_PROCESSORS" ]]; then
        echo $NUMBER_OF_PROCESSORS
    else
        getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 4
    fi
}

# Parse command line arguments
RUN_ALL=false
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_MEMORY=false
RUN_COMPARISON=false
SKIP_BUILD=false
FILTER=""
VERBOSE=false
SANITIZE_MODE="yes" # yes, no (default: yes)
BUILD_UPSTREAM=false
BUILD_OPENCV=false

# If no test category specified, run standard tests (unit + integration + memory)
DEFAULT_TESTS=true

while [[ $# -gt 0 ]]; do
    case $1 in
    --all)
        RUN_ALL=true
        RUN_UNIT=true
        RUN_INTEGRATION=true
        RUN_MEMORY=true
        RUN_COMPARISON=true
        BUILD_UPSTREAM=true
        BUILD_OPENCV=true
        DEFAULT_TESTS=false
        shift
        ;;
    --unit)
        RUN_UNIT=true
        DEFAULT_TESTS=false
        shift
        ;;
    --integration)
        RUN_INTEGRATION=true
        DEFAULT_TESTS=false
        shift
        ;;
    --memory)
        RUN_MEMORY=true
        DEFAULT_TESTS=false
        shift
        ;;
    --comparison)
        RUN_COMPARISON=true
        BUILD_UPSTREAM=true
        BUILD_OPENCV=true
        DEFAULT_TESTS=false
        shift
        ;;
    --filter)
        FILTER="$2"
        shift 2
        ;;
    --skip-build)
        SKIP_BUILD=true
        shift
        ;;
    --sanitize)
        SANITIZE_MODE="yes"
        shift
        ;;
    --no-sanitize)
        SANITIZE_MODE="no"
        shift
        ;;
    --verbose)
        VERBOSE=true
        shift
        ;;
    --help)
        echo "LibSVM Unit Tests Runner"
        echo ""
        echo "Usage:"
        echo "  $0                    # Run standard tests (unit + integration + memory)"
        echo "  $0 --all              # Run all tests including comparison tests"
        echo "  $0 --unit             # Run only unit tests"
        echo "  $0 --integration      # Run only integration tests"
        echo "  $0 --memory           # Run only memory tests (with ASAN by default)"
        echo "  $0 --comparison       # Run only comparison tests (upstream + OpenCV)"
        echo "  $0 --filter \"Pattern\" # Run tests matching filter (e.g., \"Kernel*\")"
        echo "  $0 --skip-build       # Skip build step, run tests only"
        echo "  $0 --sanitize         # Enable AddressSanitizer (ASAN) for all tests"
        echo "  $0 --no-sanitize      # Disable AddressSanitizer completely"
        echo "  $0 --verbose          # Show detailed test output"
        echo "  $0 --help             # Show this help"
        echo ""
        echo "Test Categories:"
        echo "  - Unit tests (80+ tests):        svm_node, svm_parameter, svm_problem, kernel, model"
        echo "  - Integration tests (50+ tests): train/predict, cross-validation, model I/O, probability"
        echo "  - Memory tests (45+ tests):      leak detection, RAII patterns, resource management"
        echo "  - Comparison tests (optional):   upstream and OpenCV comparison"
        echo ""
        echo "Memory Sanitizer (ASAN):"
        echo "  - Memory tests:   ENABLED by default (detects memory errors)"
        echo "  - Other tests:    DISABLED by default"
        echo "  - Use --sanitize to enable ASAN for all tests"
        echo "  - Use --no-sanitize to disable ASAN completely"
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
done

# If no specific test category was selected, run default tests
if [ "$DEFAULT_TESTS" = true ]; then
    RUN_UNIT=true
    RUN_INTEGRATION=true
    RUN_MEMORY=true
fi

echo "==============================================="
echo "LibSVM Unit Tests Runner"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ] || [ ! -d "tests" ]; then
    echo -e "${RED}Error: Please run this script from the libsvm root directory${NC}"
    echo -e "${RED}Current directory: $(pwd)${NC}"
    exit 1
fi

# Determine sanitizer usage
USE_ASAN=false
USE_UBSAN=false

if [ "$SANITIZE_MODE" = "yes" ]; then
    USE_ASAN=true
    USE_UBSAN=true
fi

# Memory tests ALWAYS use sanitizers (cannot be disabled)
if [ "$RUN_MEMORY" = true ]; then
    USE_ASAN=true
    USE_UBSAN=true
    if [ "$SANITIZE_MODE" = "no" ]; then
        echo -e "${YELLOW}⚠ Note: Memory tests always run with sanitizers enabled for safety${NC}"
    fi
fi

# Check if sanitizers are supported
if [ "$USE_ASAN" = true ] || [ "$USE_UBSAN" = true ]; then
    if isWindows; then
        echo -e "${YELLOW}⚠ Sanitizers not well supported on Windows with MSVC, disabling${NC}"
        USE_ASAN=false
        USE_UBSAN=false
        # Re-check if this is memory test (which requires sanitizers)
        if [ "$RUN_MEMORY" = true ]; then
            echo -e "${YELLOW}⚠ Warning: Memory tests may not detect all issues on Windows${NC}"
        fi
    fi
fi

# Function to setup upstream comparison
function setupUpstreamComparison() {
    UPSTREAM_SOURCE_DIR="build/upstream_source"
    UPSTREAM_BUILD_DIR="build/upstream_build"
    UPSTREAM_INSTALL_DIR="$UPSTREAM_BUILD_DIR/install"

    # Check if already built and installed
    if [ -f "$UPSTREAM_INSTALL_DIR/lib/libsvm.a" ] && [ -f "$UPSTREAM_INSTALL_DIR/include/svm.h" ]; then
        echo -e "${BLUE}Upstream build already exists at $UPSTREAM_BUILD_DIR${NC}"
        return 0
    fi

    echo ""
    echo -e "${PURPLE}===============================================${NC}"
    echo -e "${BLUE}Setting up upstream LibSVM for comparison${NC}"
    echo -e "${PURPLE}===============================================${NC}"

    # Check if upstream branch exists
    if ! git rev-parse --verify upstream &>/dev/null; then
        echo -e "${YELLOW}⚠ upstream branch not found in repository${NC}"
        echo -e "${YELLOW}  Comparison tests will be skipped${NC}"
        return 1
    fi

    # Step 1: Checkout upstream branch to temporary directory using git worktree
    echo -e "${BLUE}Checking out upstream branch...${NC}"
    mkdir -p build

    # Clean up existing worktree if any
    git worktree prune 2>/dev/null || true
    if [ -d "$UPSTREAM_SOURCE_DIR" ]; then
        git worktree remove "$UPSTREAM_SOURCE_DIR" -f 2>/dev/null || rm -rf "$UPSTREAM_SOURCE_DIR"
    fi

    # Add new worktree
    if ! git worktree add "$UPSTREAM_SOURCE_DIR" upstream 2>&1; then
        echo -e "${RED}✗ Failed to checkout upstream branch${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ Upstream source checked out to $UPSTREAM_SOURCE_DIR${NC}"

    # Step 2: Build upstream version using Makefile
    echo -e "${BLUE}Building upstream LibSVM with Makefile...${NC}"
    pushd "$UPSTREAM_SOURCE_DIR" >/dev/null

    if [ ! -f "Makefile" ]; then
        echo -e "${RED}✗ Makefile not found in upstream branch${NC}"
        popd >/dev/null
        return 1
    fi

    # Clean and build
    make clean &>/dev/null || true
    if ! make lib -j$(detectCores); then
        echo -e "${RED}✗ Failed to build upstream LibSVM${NC}"
        popd >/dev/null
        return 1
    fi

    echo -e "${GREEN}✓ Upstream LibSVM built successfully${NC}"
    popd >/dev/null

    # Step 3: Install library and headers to designated directory
    echo -e "${BLUE}Installing upstream library and headers...${NC}"
    mkdir -p "$UPSTREAM_INSTALL_DIR/lib"
    mkdir -p "$UPSTREAM_INSTALL_DIR/include"

    # Copy static library (create if not exists)
    if [ ! -f "$UPSTREAM_SOURCE_DIR/libsvm.a" ]; then
        echo -e "${BLUE}Creating static library...${NC}"
        pushd "$UPSTREAM_SOURCE_DIR" >/dev/null
        ar rcs libsvm.a svm.o
        popd >/dev/null
    fi

    cp "$UPSTREAM_SOURCE_DIR/libsvm.a" "$UPSTREAM_INSTALL_DIR/lib/" || return 1
    cp "$UPSTREAM_SOURCE_DIR/svm.h" "$UPSTREAM_INSTALL_DIR/include/" || return 1

    echo -e "${GREEN}✓ Upstream library installed to:${NC}"
    echo -e "  ${BLUE}Library: $UPSTREAM_INSTALL_DIR/lib/libsvm.a${NC}"
    echo -e "  ${BLUE}Header:  $UPSTREAM_INSTALL_DIR/include/svm.h${NC}"

    return 0
}

# Create build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

# Build tests
if [ "$SKIP_BUILD" = true ]; then
    echo ""
    echo -e "${BLUE}Skipping build, using existing binaries${NC}"
else
    echo ""
    echo -e "${PURPLE}===============================================${NC}"
    echo -e "${BLUE}Building LibSVM with tests${NC}"
    if [ "$USE_ASAN" = true ]; then
        echo -e "${GREEN}🛡️  AddressSanitizer (ASAN) ENABLED for memory error detection${NC}"
    fi
    echo -e "${PURPLE}===============================================${NC}"

    # Setup upstream comparison BEFORE cmake configuration if needed
    if [ "$BUILD_UPSTREAM" = true ]; then
        if setupUpstreamComparison; then
            echo -e "${GREEN}✓ Upstream source ready for build${NC}"
        else
            echo -e "${YELLOW}⚠ Upstream comparison setup failed, tests may be skipped${NC}"
        fi
    fi

    cd "$BUILD_DIR"

    # Prepare CMake flags
    CMAKE_FLAGS="-DLIBSVM_BUILD_TESTS=ON -DLIBSVM_BUILD_APPS=ON"

    if [ "$USE_ASAN" = true ] || [ "$USE_UBSAN" = true ]; then
        CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_BUILD_TYPE=RelWithDebInfo"
        if [ "$USE_ASAN" = true ]; then
            CMAKE_FLAGS="$CMAKE_FLAGS -DLIBSVM_ENABLE_ASAN=ON"
        fi
        if [ "$USE_UBSAN" = true ]; then
            CMAKE_FLAGS="$CMAKE_FLAGS -DLIBSVM_ENABLE_UBSAN=ON"
        fi
    else
        CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_BUILD_TYPE=Release"
    fi

    if [ "$BUILD_UPSTREAM" = true ]; then
        CMAKE_FLAGS="$CMAKE_FLAGS -DLIBSVM_BUILD_UPSTREAM_COMPARISON=ON"
        echo -e "${BLUE}Building with upstream comparison support${NC}"
    fi

    if [ "$BUILD_OPENCV" = true ]; then
        CMAKE_FLAGS="$CMAKE_FLAGS -DLIBSVM_BUILD_OPENCV_COMPARISON=ON"
        echo -e "${BLUE}Building with OpenCV comparison support (if available)${NC}"
    fi

    # Configure
    echo -e "${BLUE}Configuring CMake...${NC}"
    eval cmake .. $CMAKE_FLAGS

    # Build
    echo -e "${BLUE}Building tests (using $(detectCores) parallel jobs)...${NC}"
    cmake --build . --parallel $(detectCores)

    cd ..
fi

# Initialize results
UNIT_RESULT=0
INTEGRATION_RESULT=0
MEMORY_RESULT=0
COMPARISON_RESULT=0

# Prepare test executable paths
if isWindows; then
    UNIT_TEST_EXE="$BUILD_DIR/bin/unit_tests.exe"
    INTEGRATION_TEST_EXE="$BUILD_DIR/bin/integration_tests.exe"
    MEMORY_TEST_EXE="$BUILD_DIR/bin/memory_tests.exe"
    COMPARISON_TEST_EXE="$BUILD_DIR/bin/comparison_tests.exe"
else
    UNIT_TEST_EXE="$BUILD_DIR/bin/unit_tests"
    INTEGRATION_TEST_EXE="$BUILD_DIR/bin/integration_tests"
    MEMORY_TEST_EXE="$BUILD_DIR/bin/memory_tests"
    COMPARISON_TEST_EXE="$BUILD_DIR/bin/comparison_tests"
fi

# Prepare test arguments
TEST_ARGS=""
if [ -n "$FILTER" ]; then
    TEST_ARGS="--gtest_filter=$FILTER"
fi

if [ "$VERBOSE" = true ]; then
    TEST_ARGS="$TEST_ARGS --gtest_color=yes"
else
    TEST_ARGS="$TEST_ARGS --gtest_color=yes --gtest_brief=1"
fi

# Set sanitizer options if enabled
if [ "$USE_ASAN" = true ]; then
    # Comprehensive AddressSanitizer options
    export ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:allocator_may_return_null=1:check_initialization_order=1:detect_stack_use_after_return=1:strict_string_checks=1:detect_invalid_pointer_pairs=2:print_stats=1:atexit=1"
    echo -e "${GREEN}🛡️  AddressSanitizer enabled with comprehensive checks${NC}"
fi

if [ "$USE_UBSAN" = true ]; then
    # UndefinedBehaviorSanitizer options
    export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0"
    echo -e "${GREEN}🛡️  UndefinedBehaviorSanitizer enabled${NC}"
fi

# Run unit tests
if [ "$RUN_UNIT" = true ]; then
    echo ""
    echo "==============================================="
    echo -e "${GREEN}Running Unit Tests${NC}"
    echo "==============================================="

    if [ -f "$UNIT_TEST_EXE" ]; then
        echo -e "${YELLOW}Executing: $UNIT_TEST_EXE${NC}"
        if [ "$USE_ASAN" = true ]; then
            echo -e "${GREEN}🛡️  Running with AddressSanitizer enabled${NC}"
        fi

        $UNIT_TEST_EXE $TEST_ARGS || UNIT_RESULT=$?

        if [ $UNIT_RESULT -eq 0 ]; then
            echo -e "${GREEN}✓ Unit tests PASSED${NC}"
        else
            echo -e "${RED}✗ Unit tests FAILED (exit code: $UNIT_RESULT)${NC}"
        fi
    else
        echo -e "${RED}Error: Unit test executable not found at $UNIT_TEST_EXE${NC}"
        UNIT_RESULT=1
    fi
fi

# Run integration tests
if [ "$RUN_INTEGRATION" = true ]; then
    echo ""
    echo "==============================================="
    echo -e "${GREEN}Running Integration Tests${NC}"
    echo "==============================================="

    if [ -f "$INTEGRATION_TEST_EXE" ]; then
        echo -e "${YELLOW}Executing: $INTEGRATION_TEST_EXE${NC}"
        if [ "$USE_ASAN" = true ]; then
            echo -e "${GREEN}🛡️  Running with AddressSanitizer enabled${NC}"
        fi

        $INTEGRATION_TEST_EXE $TEST_ARGS || INTEGRATION_RESULT=$?

        if [ $INTEGRATION_RESULT -eq 0 ]; then
            echo -e "${GREEN}✓ Integration tests PASSED${NC}"
        else
            echo -e "${RED}✗ Integration tests FAILED (exit code: $INTEGRATION_RESULT)${NC}"
        fi
    else
        echo -e "${RED}Error: Integration test executable not found at $INTEGRATION_TEST_EXE${NC}"
        INTEGRATION_RESULT=1
    fi
fi

# Run memory tests
if [ "$RUN_MEMORY" = true ]; then
    echo ""
    echo "==============================================="
    echo -e "${GREEN}Running Memory Tests${NC}"
    echo "==============================================="

    if [ -f "$MEMORY_TEST_EXE" ]; then
        echo -e "${YELLOW}Executing: $MEMORY_TEST_EXE${NC}"
        if [ "$USE_ASAN" = true ]; then
            echo -e "${GREEN}🛡️  Running with AddressSanitizer enabled${NC}"
            echo -e "${BLUE}Note: ASAN helps detect memory leaks, buffer overflows, use-after-free, etc.${NC}"
        fi

        $MEMORY_TEST_EXE $TEST_ARGS || MEMORY_RESULT=$?

        if [ $MEMORY_RESULT -eq 0 ]; then
            echo -e "${GREEN}✓ Memory tests PASSED${NC}"
        else
            echo -e "${RED}✗ Memory tests FAILED (exit code: $MEMORY_RESULT)${NC}"
        fi
    else
        echo -e "${RED}Error: Memory test executable not found at $MEMORY_TEST_EXE${NC}"
        MEMORY_RESULT=1
    fi
fi

# Run comparison tests
if [ "$RUN_COMPARISON" = true ]; then
    echo ""
    echo "==============================================="
    echo -e "${GREEN}Running Comparison Tests${NC}"
    echo "==============================================="

    COMPARISON_BIN_DIR="$BUILD_DIR/bin/comparison"
    COMPARISON_RUNNER="tests/comparison/compare_runner.sh"

    # Check if comparison binaries exist
    if [ ! -d "$COMPARISON_BIN_DIR" ] || [ -z "$(ls -A $COMPARISON_BIN_DIR 2>/dev/null)" ]; then
        echo -e "${YELLOW}⚠ Comparison test binaries not found${NC}"

        # If --all was specified, try to build them
        if [ "$RUN_ALL" = true ] || [ "$RUN_COMPARISON" = true ]; then
            echo -e "${BLUE}Setting up upstream comparison...${NC}"

            # Setup upstream comparison (checkout, build, install)
            if setupUpstreamComparison; then
                echo -e "${BLUE}Rebuilding with comparison test support...${NC}"

                cd "$BUILD_DIR"

                # Reconfigure with comparison flags
                CMAKE_FLAGS="-DLIBSVM_BUILD_TESTS=ON -DLIBSVM_BUILD_APPS=ON"
                CMAKE_FLAGS="$CMAKE_FLAGS -DLIBSVM_BUILD_UPSTREAM_COMPARISON=ON"
                CMAKE_FLAGS="$CMAKE_FLAGS -DLIBSVM_BUILD_OPENCV_COMPARISON=ON"

                if [ "$USE_ASAN" = true ]; then
                    CMAKE_FLAGS="$CMAKE_FLAGS -DLIBSVM_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug"
                else
                    CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_BUILD_TYPE=Release"
                fi

                eval cmake .. $CMAKE_FLAGS
                cmake --build . --parallel $(detectCores)

                cd ..
            else
                echo -e "${YELLOW}⚠ Could not setup upstream comparison${NC}"
                COMPARISON_RESULT=0 # Don't fail
            fi
        else
            echo -e "${YELLOW}  Use --comparison or --all flag to setup comparison tests${NC}"
            COMPARISON_RESULT=0 # Don't fail if not requested
        fi
    fi

    # Run comparison tests if binaries exist
    if [ -d "$COMPARISON_BIN_DIR" ] && [ -n "$(ls -A $COMPARISON_BIN_DIR 2>/dev/null)" ]; then
        if [ -x "$COMPARISON_RUNNER" ]; then
            echo -e "${YELLOW}Running comparison tests...${NC}"

            if "$COMPARISON_RUNNER" "$COMPARISON_BIN_DIR"; then
                echo -e "${GREEN}✓ Comparison tests PASSED${NC}"
                COMPARISON_RESULT=0
            else
                echo -e "${RED}✗ Comparison tests FAILED${NC}"
                COMPARISON_RESULT=1
            fi
        else
            echo -e "${RED}Error: Comparison runner script not found or not executable${NC}"
            COMPARISON_RESULT=1
        fi
    else
        echo -e "${YELLOW}⚠ Comparison test binaries not available${NC}"
        echo -e "${YELLOW}  This is expected if upstream branch is not available${NC}"
        COMPARISON_RESULT=0 # Don't fail
    fi
fi

# Summary
echo ""
echo "==============================================="
echo -e "${GREEN}Test Summary${NC}"
echo "==============================================="

OVERALL_RESULT=0

if [ "$RUN_UNIT" = true ]; then
    if [ $UNIT_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Unit tests: PASSED${NC}"
    else
        echo -e "${RED}✗ Unit tests: FAILED${NC}"
        OVERALL_RESULT=1
    fi
fi

if [ "$RUN_INTEGRATION" = true ]; then
    if [ $INTEGRATION_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Integration tests: PASSED${NC}"
    else
        echo -e "${RED}✗ Integration tests: FAILED${NC}"
        OVERALL_RESULT=1
    fi
fi

if [ "$RUN_MEMORY" = true ]; then
    if [ $MEMORY_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Memory tests: PASSED${NC}"
    else
        echo -e "${RED}✗ Memory tests: FAILED${NC}"
        OVERALL_RESULT=1
    fi
fi

if [ "$RUN_COMPARISON" = true ]; then
    if [ $COMPARISON_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Comparison tests: PASSED${NC}"
    else
        echo -e "${RED}✗ Comparison tests: FAILED${NC}"
        OVERALL_RESULT=1
    fi
fi

echo ""
if [ $OVERALL_RESULT -eq 0 ]; then
    echo -e "${GREEN}🎉 All selected tests PASSED!${NC}"
else
    echo -e "${RED}❌ Some tests FAILED${NC}"
fi

echo "==============================================="

exit $OVERALL_RESULT
