# LibSVM Development Log

This document records major changes and developments from the original libsvm codebase.

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

**Build System**
- Removed all Makefiles (root, java/, matlab/, python/, svm-toy/)
- Created CMake configuration with cross-platform support (Linux, macOS, Windows)
- Added build options: OpenMP, shared/static libraries, language bindings
- Implemented proper installation rules and `find_package()` support

**Directory Restructure**
- `src/`: Core library (svm.cpp, svm.h, svm.def)
- `apps/`: Command-line tools (svm-train, svm-predict, svm-scale)
- `examples/`: Sample programs and data (svm-toy Qt GUI, heart_scale dataset)
- `bindings/`: Language bindings (Python, Java, MATLAB) with CMake build
- `cmake/`: CMake modules and configuration templates
- `docs/`: Documentation and development logs

**Cleanup**
- Removed `windows/` directory with all precompiled binaries (.exe, .dll, .mexw64)
- Removed `svm-toy/windows/` version (kept Qt version only)
- Removed generated files from version control

**Language Bindings**
- Python: Updated library loading paths in `svm.py` for new structure
- Java: m4 preprocessing integrated into CMake, JAR packaging support
- MATLAB: MEX file compilation via CMake with fallback to make.m

### Migration Details

See [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for complete migration planning and task tracking.


