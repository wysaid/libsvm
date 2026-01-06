# Windows CI Simulation Script
# This script simulates the GitHub Actions CI workflow for Windows

param(
    [switch]$SkipJava = $false,
    [switch]$SkipPython = $false
)

$ErrorActionPreference = 'Stop'

# Script directory and root directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT_DIR = Split-Path -Parent $SCRIPT_DIR

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "LibSVM Windows CI Simulation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

function Pass {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Fail {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
    exit 1
}

function Info {
    param([string]$Message)
    Write-Host "→ $Message" -ForegroundColor Yellow
}

function Cleanup {
    Info "Cleaning up build directories..."
    Set-Location $ROOT_DIR
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue }
    if (Test-Path "install") { Remove-Item -Recurse -Force "install" -ErrorAction SilentlyContinue }
    if (Test-Path "test_project") { Remove-Item -Recurse -Force "test_project" -ErrorAction SilentlyContinue }
}

# ============================================================================
# Job 1: Build and Test (Default Compiler, Release + Debug)
# ============================================================================
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Job 1: Build and Test (Default Compiler)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

foreach ($BUILD_TYPE in @("Release", "Debug")) {
    Info "Testing with BUILD_TYPE=$BUILD_TYPE"
    Cleanup

    # Configure
    Info "Configuring CMake..."
    cmake -B "$ROOT_DIR/build" `
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE `
        -DLIBSVM_BUILD_APPS=ON `
        -DLIBSVM_BUILD_EXAMPLES=OFF

    # Build
    Info "Building..."
    cmake --build "$ROOT_DIR/build" --config $BUILD_TYPE

    # Test svm-train
    Info "Testing svm-train..."
    & "$ROOT_DIR/build/bin/$BUILD_TYPE/svm-train.exe" `
        "$ROOT_DIR/examples/data/heart_scale" `
        "$ROOT_DIR/build/heart_scale.model"

    # Test svm-predict
    Info "Testing svm-predict..."
    & "$ROOT_DIR/build/bin/$BUILD_TYPE/svm-predict.exe" `
        "$ROOT_DIR/examples/data/heart_scale" `
        "$ROOT_DIR/build/heart_scale.model" `
        "$ROOT_DIR/build/heart_scale.output"

    # Test svm-scale
    Info "Testing svm-scale..."
    & "$ROOT_DIR/build/bin/$BUILD_TYPE/svm-scale.exe" `
        -l -1 -u 1 `
        "$ROOT_DIR/examples/data/heart_scale" `
        | Out-File -FilePath "$ROOT_DIR/build/heart_scale.scaled"

    # Verify outputs
    Info "Verifying outputs..."
    if (!(Test-Path "$ROOT_DIR/build/heart_scale.model")) {
        Fail "Model file not created"
    }
    if (!(Test-Path "$ROOT_DIR/build/heart_scale.output")) {
        Fail "Prediction output not created"
    }

    Pass "Build and test completed for $BUILD_TYPE"
}

Cleanup

# ============================================================================
# Job 2: Shared Library Build
# ============================================================================
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Job 2: Shared Library Build" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Info "Configuring CMake for shared library..."
cmake -B "$ROOT_DIR/build" `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_SHARED_LIBS=ON `
    -DLIBSVM_BUILD_APPS=ON

Info "Building..."
cmake --build "$ROOT_DIR/build" --config Release

Info "Verifying shared library..."
if (Test-Path "$ROOT_DIR/build/bin/Release/svm.dll") {
    Pass "Shared library built successfully (Release/svm.dll)"
} elseif (Test-Path "$ROOT_DIR/build/bin/svm.dll") {
    Pass "Shared library built successfully (svm.dll)"
} else {
    Fail "Shared library not found"
}

Cleanup

# ============================================================================
# Job 3: Python Bindings
# ============================================================================
if (!$SkipPython) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Job 3: Python Bindings" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan

    if (Get-Command python -ErrorAction SilentlyContinue) {
        Info "Building shared library for Python..."
        cmake -B "$ROOT_DIR/build" `
            -DCMAKE_BUILD_TYPE=Release `
            -DBUILD_SHARED_LIBS=ON `
            -DLIBSVM_BUILD_PYTHON=ON
        cmake --build "$ROOT_DIR/build" --config Release

        Info "Copying library to Python package..."
        if (Test-Path "$ROOT_DIR/build/bin/Release/svm.dll") {
            Copy-Item "$ROOT_DIR/build/bin/Release/svm.dll" "$ROOT_DIR/bindings/python/libsvm/"
        } elseif (Test-Path "$ROOT_DIR/build/bin/svm.dll") {
            Copy-Item "$ROOT_DIR/build/bin/svm.dll" "$ROOT_DIR/bindings/python/libsvm/"
        } else {
            Fail "svm.dll not found"
        }

        Info "Testing Python import..."
        Set-Location "$ROOT_DIR/bindings/python"
        python -c "from libsvm import svmutil; print('Python bindings import successful')"

        Info "Testing Python training..."
        python -c @"
from libsvm import svmutil
import sys
sys.path.insert(0, '.')
y, x = svmutil.svm_read_problem('../../examples/data/heart_scale')
prob = svmutil.svm_problem(y[:200], x[:200])
param = svmutil.svm_parameter('-q')
model = svmutil.svm_train(prob, param)
print('Python training test successful')
"@
        Set-Location $ROOT_DIR
        Pass "Python bindings test completed"
        Cleanup
    } else {
        Write-Host "⚠ Python not found, skipping Python bindings test" -ForegroundColor Yellow
    }
}

# ============================================================================
# Job 4: Java Bindings (requires m4)
# ============================================================================
if (!$SkipJava) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Job 4: Java Bindings" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan

    $javaAvailable = Get-Command java -ErrorAction SilentlyContinue
    $m4Path = $null
    
    # Check for m4 in various locations
    $m4Locations = @(
        "C:\tools\msys64\usr\bin\m4.exe",
        "C:\msys64\usr\bin\m4.exe",
        "C:\Program Files\Git\usr\bin\m4.exe"
    )
    
    foreach ($loc in $m4Locations) {
        if (Test-Path $loc) {
            $m4Path = $loc
            Info "Found m4 at: $m4Path"
            break
        }
    }

    if ($javaAvailable -and $m4Path) {
        Info "Configuring CMake for Java bindings..."
        $m4PathForCMake = $m4Path -replace '\\', '/'
        
        # Add m4 to PATH
        $env:PATH = "$(Split-Path $m4Path);" + $env:PATH
        
        cmake -B "$ROOT_DIR/build" `
            -DCMAKE_BUILD_TYPE=Release `
            -DLIBSVM_BUILD_JAVA=ON `
            "-DM4_EXECUTABLE=$m4PathForCMake"

        Info "Building Java bindings..."
        cmake --build "$ROOT_DIR/build" --target java-bindings --config Release

        Info "Testing Java bindings..."
        java -classpath "$ROOT_DIR/build/bindings/java/libsvm.jar" svm_train `
            "$ROOT_DIR/examples/data/heart_scale" `
            "$ROOT_DIR/build/heart_scale_java.model"

        Pass "Java bindings test completed"
        Cleanup
    } else {
        if (!$javaAvailable) {
            Write-Host "⚠ Java not found, skipping Java bindings test" -ForegroundColor Yellow
        }
        if (!$m4Path) {
            Write-Host "⚠ m4 not found in standard locations, skipping Java bindings test" -ForegroundColor Yellow
            Write-Host "  You can install m4 via MSYS2:" -ForegroundColor Yellow
            Write-Host "  1. Install MSYS2: choco install msys2 (requires admin)" -ForegroundColor Yellow
            Write-Host "  2. Run: C:\tools\msys64\usr\bin\bash.exe -lc 'pacman -S --noconfirm m4'" -ForegroundColor Yellow
        }
    }
}

# ============================================================================
# Job 5: Installation Test
# ============================================================================
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Job 5: Installation Test" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Info "Configuring CMake for installation..."
cmake -B "$ROOT_DIR/build" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$ROOT_DIR/install" `
    -DLIBSVM_BUILD_APPS=ON

Info "Building..."
cmake --build "$ROOT_DIR/build" --config Release

Info "Installing..."
cmake --install "$ROOT_DIR/build" --config Release

Info "Verifying installation..."
if (!(Test-Path "$ROOT_DIR/install/include/svm.h")) { Fail "svm.h not installed" }
if (!(Test-Path "$ROOT_DIR/install/bin/svm-train.exe")) { Fail "svm-train.exe not installed" }
if (!(Test-Path "$ROOT_DIR/install/bin/svm-predict.exe")) { Fail "svm-predict.exe not installed" }
if (!(Test-Path "$ROOT_DIR/install/bin/svm-scale.exe")) { Fail "svm-scale.exe not installed" }

Info "Testing find_package..."
New-Item -ItemType Directory -Force -Path "$ROOT_DIR/test_project" | Out-Null

@"
cmake_minimum_required(VERSION 3.16)
project(TestLibSVM)
find_package(LibSVM REQUIRED)
add_executable(test_app test.cpp)
target_link_libraries(test_app LibSVM::svm)
"@ | Out-File -FilePath "$ROOT_DIR/test_project/CMakeLists.txt" -Encoding utf8

@"
#include <svm.h>
int main() { return 0; }
"@ | Out-File -FilePath "$ROOT_DIR/test_project/test.cpp" -Encoding utf8

cmake -B "$ROOT_DIR/test_project/build" `
    -S "$ROOT_DIR/test_project" `
    -DCMAKE_PREFIX_PATH="$ROOT_DIR/install"

cmake --build "$ROOT_DIR/test_project/build" --config Release

Pass "Installation test completed"

# ============================================================================
# Final Cleanup
# ============================================================================
Cleanup

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✓ All CI jobs completed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
