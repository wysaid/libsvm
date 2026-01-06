# ============================================================================
# LibSVM Build Options
# ============================================================================
#
# This file contains additional build option configurations.
# Include this file if you want to customize build settings.
#
# ============================================================================

# ============================================================================
# Compiler-specific settings
# ============================================================================

# GCC/Clang specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    # Enable additional optimizations for Release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Consider enabling these for better performance:
        # -march=native    : Optimize for local CPU architecture
        # -flto            : Link-time optimization
        # -funroll-loops   : Unroll loops for better performance
        
        # Uncomment to enable:
        # add_compile_options(-march=native)
        # add_compile_options(-flto)
    endif()
    
    # Debug build settings
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    endif()
endif()

# MSVC specific settings
if(MSVC)
    # Enable parallel compilation
    add_compile_options(/MP)
    
    # Disable specific warnings
    add_compile_options(/wd4244 /wd4267)  # Conversion warnings
endif()

# ============================================================================
# Platform-specific settings
# ============================================================================

# macOS specific
if(APPLE)
    # Set deployment target if not specified
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum macOS version")
    endif()
endif()

# Windows specific
if(WIN32)
    # Ensure proper DLL export/import
    if(BUILD_SHARED_LIBS)
        add_definitions(-DLIBSVM_EXPORTS)
    endif()
endif()
