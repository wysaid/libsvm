#!/bin/bash
# Test Java bindings build with Git Bash (simulating CI environment)

set -e

echo "=========================================="
echo "Testing Java Bindings with Git Bash"
echo "=========================================="
echo ""

# Check if m4 is in PATH (simulating MSYS2 installation)
M4_PATHS=(
    "/c/tools/msys64/usr/bin"
    "/c/msys64/usr/bin"
)

M4_FOUND=""
for M4_PATH in "${M4_PATHS[@]}"; do
    if [ -f "$M4_PATH/m4.exe" ]; then
        echo "✓ Found m4 at: $M4_PATH"
        export PATH="$M4_PATH:$PATH"
        M4_FOUND="yes"
        break
    fi
done

if [ -z "$M4_FOUND" ]; then
    echo "⚠ m4 not found in standard locations"
    echo "  To install: choco install msys2 (requires admin)"
    echo "  Then run: /c/tools/msys64/usr/bin/bash.exe -lc 'pacman -S --noconfirm m4'"
    exit 1
fi

# Check Java
if ! command -v java &>/dev/null; then
    echo "✗ Java not found"
    exit 1
fi

echo "✓ Java found: $(java -version 2>&1 | head -n 1)"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Clean up
echo "→ Cleaning build directory..."
rm -rf "$ROOT_DIR/build"

# Configure
echo "→ Configuring CMake for Java bindings..."
cmake -B "$ROOT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBSVM_BUILD_JAVA=ON

# Build
echo "→ Building Java bindings..."
cmake --build "$ROOT_DIR/build" --target java-bindings --config Release

# Test
echo "→ Testing Java bindings..."
java -classpath "$ROOT_DIR/build/bindings/java/libsvm.jar" svm_train \
    "$ROOT_DIR/examples/data/heart_scale" \
    "$ROOT_DIR/build/heart_scale_java.model"

echo ""
echo "=========================================="
echo "✓ Java bindings test completed!"
echo "=========================================="
