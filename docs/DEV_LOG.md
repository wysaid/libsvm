# LibSVM Development Log

Records major changes from the upstream libsvm repository.

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


