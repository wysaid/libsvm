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
# Memory Sanitizer (ASAN):
# - Enabled by default for memory tests
# - Can be enabled/disabled with --sanitize/--no-sanitize
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

set -e  # Exit on any error

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
SANITIZE_MODE="auto"  # auto, yes, no
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
NC='\033[0m'  # No Color

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ] || [ ! -d "tests" ]; then
    echo -e "${RED}Error: Please run this script from the libsvm root directory${NC}"
    echo -e "${RED}Current directory: $(pwd)${NC}"
    exit 1
fi

# Determine ASAN usage
USE_ASAN=false
if [ "$SANITIZE_MODE" = "yes" ]; then
    USE_ASAN=true
elif [ "$SANITIZE_MODE" = "auto" ]; then
    # Auto mode: enable ASAN only for memory tests
    if [ "$RUN_MEMORY" = true ] && [ "$RUN_UNIT" = false ] && [ "$RUN_INTEGRATION" = false ] && [ "$RUN_COMPARISON" = false ]; then
        USE_ASAN=true
    fi
fi

# Check if ASAN is supported
if [ "$USE_ASAN" = true ]; then
    if isWindows; then
        echo -e "${YELLOW}⚠ AddressSanitizer not well supported on Windows with MSVC, disabling ASAN${NC}"
        USE_ASAN=false
    fi
fi

# Function to setup upstream comparison
function setupUpstreamComparison() {
    UPSTREAM_DIR="tests/upstream_libsvm"
    
    if [ -d "$UPSTREAM_DIR" ] && [ -f "$UPSTREAM_DIR/svm.cpp" ]; then
        echo -e "${BLUE}Upstream source already exists at $UPSTREAM_DIR${NC}"
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
    
    # Create upstream directory
    echo -e "${BLUE}Extracting upstream branch to $UPSTREAM_DIR...${NC}"
    rm -rf "$UPSTREAM_DIR"
    mkdir -p "$UPSTREAM_DIR"
    
    # Use git show to extract files from upstream branch
    # Get list of files from upstream branch
    FILES=$(git ls-tree -r --name-only upstream)
    
    # Extract each file
    EXTRACT_COUNT=0
    for file in $FILES; do
        # Create directory structure
        FILE_DIR=$(dirname "$file")
        mkdir -p "$UPSTREAM_DIR/$FILE_DIR"
        
        # Extract file content
        if git show "upstream:$file" > "$UPSTREAM_DIR/$file" 2>/dev/null; then
            ((EXTRACT_COUNT++))
        fi
    done
    
    if [ $EXTRACT_COUNT -gt 0 ] && [ -f "$UPSTREAM_DIR/svm.cpp" ]; then
        echo -e "${GREEN}✓ Upstream LibSVM source extracted ($EXTRACT_COUNT files)${NC}"
        echo -e "${BLUE}The comparison tests will be built automatically by CMake${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to extract upstream source${NC}"
        return 1
    fi
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
    
    if [ "$USE_ASAN" = true ]; then
        CMAKE_FLAGS="$CMAKE_FLAGS -DLIBSVM_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug"
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

# Set ASAN options if enabled
if [ "$USE_ASAN" = true ]; then
    export ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:allocator_may_return_null=1"
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
    
    # Check if comparison_tests executable exists
    if [ ! -f "$COMPARISON_TEST_EXE" ]; then
        echo -e "${YELLOW}⚠ Comparison test executable not found at $COMPARISON_TEST_EXE${NC}"
        
        # If --all was specified, try to build it
        if [ "$RUN_ALL" = true ]; then
            echo -e "${BLUE}Attempting to setup and build comparison tests...${NC}"
            
            # Setup upstream comparison
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
                COMPARISON_RESULT=0  # Don't fail
            fi
        else
            echo -e "${YELLOW}  Use --all flag to automatically setup comparison tests${NC}"
            COMPARISON_RESULT=0  # Don't fail if comparison tests weren't built
        fi
    fi
    
    if [ -f "$COMPARISON_TEST_EXE" ]; then
        echo -e "${YELLOW}Executing: $COMPARISON_TEST_EXE${NC}"
        echo -e "${BLUE}Note: Some tests may be skipped if dependencies (upstream/OpenCV) are not available${NC}"
        
        $COMPARISON_TEST_EXE $TEST_ARGS || COMPARISON_RESULT=$?
        
        if [ $COMPARISON_RESULT -eq 0 ]; then
            echo -e "${GREEN}✓ Comparison tests PASSED${NC}"
        else
            echo -e "${RED}✗ Comparison tests FAILED (exit code: $COMPARISON_RESULT)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Comparison test executable still not found${NC}"
        echo -e "${YELLOW}  This is expected if dependencies are not available${NC}"
        COMPARISON_RESULT=0  # Don't fail if comparison tests couldn't be built
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
