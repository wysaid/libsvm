# Fork Changes from Upstream

This document outlines major modifications made in this fork compared to the original [cjlin1/libsvm](https://github.com/cjlin1/libsvm).

**Purpose**: Track significant architectural changes, not day-to-day development details.

---

## 2026-01-06: C++17 Modernization and Code Quality Improvements

Upgraded codebase to C++17 standard with modern C++ practices.

**Language Standard**
- Upgraded from C++11 to C++17
- Added `CMAKE_CXX_EXTENSIONS OFF` to enforce standard compliance
- Cross-platform compiler support (GCC, Clang, MSVC, AppleClang)

**Windows Platform Improvements**
- Added `NOMINMAX` to prevent Windows.h min/max macro conflicts
- Added `WIN32_LEAN_AND_MEAN` to reduce unnecessary includes
- Added `_CRT_SECURE_NO_WARNINGS` for legacy C functions
- Added `/utf-8` flag for MSVC to ensure proper source encoding

**Code Modernization**
- Removed custom `min()`, `max()`, `swap()` templates - now using `std::min`, `std::max`, `std::swap` from `<algorithm>`
- Added C++ standard library headers: `<algorithm>`, `<memory>`, `<vector>`
- Improved `clone()` template function with explicit `static_cast<T>()` for type-safe conversions
- Added RAII `auto_array<T>` wrapper class for automatic memory management

**Memory Management Improvements**
- Replaced manual `new`/`delete[]` pairs with `std::vector` in key solver functions:
  - `solve_c_svc()`: vector for minus_ones, y arrays (with direct initialization)
  - `solve_nu_svc()`: vector for y, zeros arrays
  - `solve_one_class()`: vector for zeros, ones arrays (with direct initialization)
  - `solve_epsilon_svr()`: vector for alpha2, linear_term, y arrays
  - `solve_nu_svr()`: vector for alpha2, linear_term, y arrays
- Uses `std::vector::data()` for C API compatibility
- Automatic memory management through RAII - prevents memory leaks from early returns or exceptions
- Convenient initialization with size and default values (e.g., `vector<double>(l, -1.0)`)
- Legacy `Malloc()` macro retained for backward compatibility in other parts of codebase

**Benefits**
- Improved type safety and reduced implicit conversion warnings
- Better memory safety through RAII patterns
- Enhanced cross-platform compatibility
- Modern C++ idioms without breaking API compatibility

---

## 2026-01-06: CMake Build System Migration

Migrated from Makefile to modern CMake (v3.16+) with cross-platform support.

**Build System**
- CMake configuration for Linux, macOS, Windows
- Build options: OpenMP, shared/static libraries, language bindings
- Proper installation and `find_package()` support

**Directory Structure**
- `src/`: Core library (svm.cpp, svm.h, svm.def)
- `apps/`: Command-line tools (svm-train, svm-predict, svm-scale)
- `examples/`: Sample programs (svm-toy Qt GUI, heart_scale data)
- `bindings/`: Python, Java, MATLAB with CMake integration
- `cmake/`: CMake modules and config templates

**Cleanup**
- Removed precompiled Windows binaries
- Removed Makefiles and generated files

**CI/CD**
- GitHub Actions workflow for multi-platform testing
- Python 3.9/3.11/3.13, Java 11/17/21 support
- Static analysis, code coverage, installation tests

**Key Fixes**
- Windows: Set `MSYS_NO_PATHCONV=1` to prevent Git Bash path corruption
- Ubuntu: Updated to gcc-13/clang-18 for ubuntu-latest (24.04)
- CMake Export: Added `INCLUDES DESTINATION` for proper `find_package()` support
- Shell: Standardized on bash across all platforms
