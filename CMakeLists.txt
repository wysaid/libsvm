# ============================================================================
# LibSVM - A Library for Support Vector Machines
# ============================================================================
# CMake build configuration
# Minimum CMake version: 3.16
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# Extract version from svm.h
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/svm.h" SVM_H_CONTENT)
string(REGEX MATCH "#define LIBSVM_VERSION ([0-9]+)" _ ${SVM_H_CONTENT})
set(LIBSVM_VERSION_RAW ${CMAKE_MATCH_1})
math(EXPR LIBSVM_VERSION_MAJOR "${LIBSVM_VERSION_RAW} / 100")
math(EXPR LIBSVM_VERSION_MINOR "${LIBSVM_VERSION_RAW} % 100")
set(LIBSVM_VERSION "${LIBSVM_VERSION_MAJOR}.${LIBSVM_VERSION_MINOR}")

project(libsvm
    VERSION ${LIBSVM_VERSION}
    DESCRIPTION "A Library for Support Vector Machines"
    HOMEPAGE_URL "https://www.csie.ntu.edu.tw/~cjlin/libsvm/"
    LANGUAGES C CXX
)

message(STATUS "Configuring LibSVM version ${LIBSVM_VERSION}")

# ============================================================================
# Build Options
# ============================================================================

option(BUILD_SHARED_LIBS "Build shared library instead of static" ON)
option(LIBSVM_ENABLE_OPENMP "Enable OpenMP for parallel training" OFF)
option(LIBSVM_BUILD_APPS "Build command-line tools (svm-train, svm-predict, svm-scale)" ON)
option(LIBSVM_BUILD_EXAMPLES "Build example programs (svm-toy)" OFF)
option(LIBSVM_BUILD_PYTHON "Build Python bindings" OFF)
option(LIBSVM_BUILD_JAVA "Build Java bindings" OFF)
option(LIBSVM_BUILD_MATLAB "Build MATLAB/MEX bindings" OFF)

# ============================================================================
# Compiler Settings
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Position Independent Code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Platform-specific definitions to avoid conflicts
if(WIN32)
    # Prevent Windows.h from defining min/max macros
    add_compile_definitions(NOMINMAX)
    # Prevent Windows.h from defining unnecessary macros
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
    # Use modern secure CRT functions
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wconversion)
elseif(MSVC)
    add_compile_options(/W3)
    # Enable UTF-8 source file encoding
    add_compile_options(/utf-8)
endif()

# Optimization for Release builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        add_compile_options(-O3)
    endif()
endif()

# ============================================================================
# OpenMP Support
# ============================================================================

if(LIBSVM_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found - parallel training enabled")
    else()
        message(WARNING "OpenMP not found - parallel training disabled")
        set(LIBSVM_ENABLE_OPENMP OFF)
    endif()
endif()

# ============================================================================
# Output Directories
# ============================================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================================================
# Subdirectories
# ============================================================================

# Core library
add_subdirectory(src)

# Command-line tools
if(LIBSVM_BUILD_APPS)
    add_subdirectory(apps)
endif()

# Examples
if(LIBSVM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Language bindings
add_subdirectory(bindings)

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Generate and install CMake config files
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/LibSVMConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LibSVMConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibSVM
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/LibSVMConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/LibSVMConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/LibSVMConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibSVM
)

# Export targets
install(EXPORT LibSVMTargets
    FILE LibSVMTargets.cmake
    NAMESPACE LibSVM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibSVM
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "LibSVM ${LIBSVM_VERSION} Configuration Summary:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared library:    ${BUILD_SHARED_LIBS}")
message(STATUS "  OpenMP:            ${LIBSVM_ENABLE_OPENMP}")
message(STATUS "  Build apps:        ${LIBSVM_BUILD_APPS}")
message(STATUS "  Build examples:    ${LIBSVM_BUILD_EXAMPLES}")
message(STATUS "  Python bindings:   ${LIBSVM_BUILD_PYTHON}")
message(STATUS "  Java bindings:     ${LIBSVM_BUILD_JAVA}")
message(STATUS "  MATLAB bindings:   ${LIBSVM_BUILD_MATLAB}")
message(STATUS "")
