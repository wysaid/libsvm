#!/bin/bash
# Comparison runner script
# Runs both current and upstream versions of test cases and compares output

# Don't use set -e to see all test results
# set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <comparison_bin_dir>"
    exit 1
fi

BIN_DIR="$1"
OUTPUT_DIR="$(mktemp -d)"
trap "rm -rf $OUTPUT_DIR" EXIT

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "==============================================="
echo "Running Upstream Comparison Tests"
echo "==============================================="
echo ""

# Find all test cases
TEST_NAMES=$(ls "$BIN_DIR"/compare_current_* 2>/dev/null | sed 's/.*compare_current_//' || true)

if [ -z "$TEST_NAMES" ]; then
    echo -e "${YELLOW}No comparison tests found${NC}"
    exit 0
fi

TOTAL=0
PASSED=0
FAILED=0

for TEST_NAME in $TEST_NAMES; do
    CURRENT_EXE="$BIN_DIR/compare_current_$TEST_NAME"
    UPSTREAM_EXE="$BIN_DIR/compare_upstream_$TEST_NAME"
    
    if [ ! -x "$CURRENT_EXE" ] || [ ! -x "$UPSTREAM_EXE" ]; then
        echo -e "${YELLOW}⚠ Skipping $TEST_NAME: executable not found${NC}"
        continue
    fi
    
    ((TOTAL++))
    
    echo -n "Testing $TEST_NAME... "
    
    # Run both versions
    CURRENT_OUT="$OUTPUT_DIR/current_$TEST_NAME.txt"
    UPSTREAM_OUT="$OUTPUT_DIR/upstream_$TEST_NAME.txt"
    
    if ! "$CURRENT_EXE" > "$CURRENT_OUT" 2>&1; then
        echo -e "${RED}FAIL (current version crashed)${NC}"
        ((FAILED++))
        continue
    fi
    
    if ! "$UPSTREAM_EXE" > "$UPSTREAM_OUT" 2>&1; then
        echo -e "${RED}FAIL (upstream version crashed)${NC}"
        ((FAILED++))
        continue
    fi
    
    # Compare outputs
    if diff -q "$CURRENT_OUT" "$UPSTREAM_OUT" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}FAIL (output mismatch)${NC}"
        echo ""
        echo "  Current output:"
        sed 's/^/    /' "$CURRENT_OUT" | head -20
        echo ""
        echo "  Upstream output:"
        sed 's/^/    /' "$UPSTREAM_OUT" | head -20
        echo ""
        ((FAILED++))
    fi
done

echo ""
echo "==============================================="
echo "Comparison Test Summary"
echo "==============================================="
echo "Total:  $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
fi
echo ""

if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
