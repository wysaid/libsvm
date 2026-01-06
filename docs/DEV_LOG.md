# LibSVM Development Log

This document records major changes and developments from the original libsvm codebase.

---

## 2026-01-06: Critical CI Fixes - Path Conversion, Compilers, and CMake Export

Fixed three critical issues preventing CI from passing.

#### Issue 1: Windows Git Bash Path Conversion
**Problem**: Git Bash on Windows automatically converts Unix-like paths, causing CMake errors:
- Input: `D:/a/libsvm/libsvm`
- Converted to: `D:alibsvmlibsvm` (slashes removed)
- Error: `The source directory "D:alibsvmlibsvm/build/..." does not exist`

**Root Cause**: MSYS2 automatic path conversion in Git Bash for Windows.

**Solution**: Added `MSYS_NO_PATHCONV: 1` to workflow environment variables to disable automatic path conversion.

#### Issue 2: Ubuntu Compiler Version Mismatch
**Problem**: Configured compilers (`gcc-11`, `clang-14`) don't exist on ubuntu-latest (24.04):
- Error: `Could not find compiler set in environment variable CC: gcc-11`

**Solution**: Updated to compilers available on Ubuntu 24.04:
- `gcc-11` → `gcc-13`
- `clang-14` → `clang-18`

#### Issue 3: CMake Export Missing Include Directories
**Problem**: `find_package(LibSVM)` succeeds but compilation fails:
- Error: `fatal error: svm.h: No such file or directory`
- Root cause: Generated `LibSVMTargets.cmake` lacked `INTERFACE_INCLUDE_DIRECTORIES` property

**Analysis**: The `install(TARGETS)` command wasn't exporting include directory information to the imported target.

**Solution**: Added `INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}` to `install(TARGETS)` in `src/CMakeLists.txt`. This directive:
1. Automatically populates `INTERFACE_INCLUDE_DIRECTORIES` in generated `LibSVMTargets.cmake`
2. Ensures downstream projects using `target_link_libraries(LibSVM::svm)` get correct include paths

**Verification**: Tested locally with installation and `find_package()` - compilation successful.

---

## 2026-01-06: CI/CD Workflow Shell Configuration

Fixed workflow failures by standardizing shell usage across all CI jobs.

#### Root Cause
Windows runners were using PowerShell by default, which incorrectly parsed multi-line commands with backslash line continuations. This caused errors like:
- `The source directory "/" does not appear to contain CMakeLists.txt`
- `-DCMAKE_BUILD_TYPE=Release is not recognized as a cmdlet`

#### Solution
Added `shell: bash` to all run steps across all jobs to ensure consistent behavior across Windows, macOS, and Linux. Bash properly handles line continuations with backslashes.

---

## 2026-01-06: CI/CD Workflow and CMake Fixes

Fixed multiple issues identified in GitHub Actions CI and code review.

#### GitHub Actions Workflow Fixes
- Fixed Windows binary paths for multi-config Visual Studio generator (added Release/Debug subdirectories)
- Improved shared library copy patterns to handle both versioned and unversioned filenames
- Set CMAKE_PREFIX_PATH for macOS Qt@5 (keg-only Homebrew formula)
- Changed cppcheck to use --error-exitcode=1 with continue-on-error for visibility

#### CMake Configuration Improvements
- Changed OpenMP linkage from PUBLIC to PRIVATE (implementation detail only)
- Removed redundant include_directories() in apps/ (inherited from svm target)
- Added defensive check for INTERFACE_INCLUDE_DIRECTORIES in config file
- Added Java source dependencies to java-bindings target for proper rebuild tracking

#### Documentation
- Converted bold emphasis to proper Markdown heading syntax (#### level) in DEV_LOG.md

---

## 2026-01-06: Documentation Reorganization

Reorganized documentation structure for better maintainability:

- Renamed `README` to `README.original` to preserve original documentation
- `README.md` now serves as primary documentation with modern Markdown format
- Added "About This Fork" section explaining modernization changes
- Established this development log to track major changes

---

## 2026-01-06: CMake Build System Migration

Migrated entire project from Makefile-based build system to modern CMake (v3.16+).

### Major Changes

#### Build System
- Removed all Makefiles (root, java/, matlab/, python/, svm-toy/)
- Created CMake configuration with cross-platform support (Linux, macOS, Windows)
- Added build options: OpenMP, shared/static libraries, language bindings
- Implemented proper installation rules and `find_package()` support

#### Directory Restructure
- `src/`: Core library (svm.cpp, svm.h, svm.def)
- `apps/`: Command-line tools (svm-train, svm-predict, svm-scale)
- `examples/`: Sample programs and data (svm-toy Qt GUI, heart_scale dataset)
- `bindings/`: Language bindings (Python, Java, MATLAB) with CMake build
- `cmake/`: CMake modules and configuration templates
- `docs/`: Documentation and development logs

#### Cleanup
- Removed `windows/` directory with all precompiled binaries (.exe, .dll, .mexw64)
- Removed `svm-toy/windows/` version (kept Qt version only)
- Removed generated files from version control

#### Language Bindings
- Python: Updated library loading paths in `svm.py` for new structure
- Java: m4 preprocessing integrated into CMake, JAR packaging support
- MATLAB: MEX file compilation via CMake with fallback to make.m

---

## 2026-01-06: CI/CD Workflow Implementation

Added comprehensive GitHub Actions workflow for continuous integration and testing.

#### Coverage
- Multi-platform builds: Ubuntu, macOS, Windows
- Multiple compilers: GCC, Clang, MSVC (default per platform)
- Build configurations: Release and Debug modes
- OpenMP support testing on Linux and macOS
- Shared and static library builds

#### Language Bindings Testing
- Python: Tests with Python 3.9, 3.11, 3.13 across all platforms
- Java: Tests with Java 11, 17, 21 across all platforms
- MATLAB: MEX file compilation support (when available)

#### Quality Assurance
- Static analysis with cppcheck
- Installation and `find_package()` verification
- Code coverage reporting with lcov and Codecov
- Qt example (svm-toy) build verification

#### Validation
- Functional tests using heart_scale dataset
- Command-line tools (svm-train, svm-predict, svm-scale) execution tests
- Output file verification


