# ============================================================================
# LibSVM Python Bindings
# ============================================================================
#
# The Python bindings use ctypes to load the libsvm shared library.
# This CMake file ensures the shared library is built and optionally
# copies it to the Python package directory for easier access.
#
# Usage:
#   cmake -DLIBSVM_BUILD_PYTHON=ON -DBUILD_SHARED_LIBS=ON ..
#   cmake --build .
#   
# After building, you can use the Python bindings:
#   export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH  # Linux
#   export DYLD_LIBRARY_PATH=/path/to/build/lib:$DYLD_LIBRARY_PATH  # macOS
#   cd bindings/python && python -c "from libsvm import svmutil"
#
# ============================================================================

message(STATUS "Configuring Python bindings")

# Ensure shared library is built (required for ctypes)
if(NOT BUILD_SHARED_LIBS)
    message(WARNING "Python bindings require BUILD_SHARED_LIBS=ON. "
                    "The shared library will be needed at runtime.")
endif()

# ============================================================================
# Find Python (optional, for installation)
# ============================================================================

find_package(Python3 COMPONENTS Interpreter)

# ============================================================================
# Copy shared library to Python package directory (optional)
# ============================================================================

option(LIBSVM_PYTHON_COPY_LIB "Copy libsvm shared library to Python package directory" ON)

if(LIBSVM_PYTHON_COPY_LIB AND BUILD_SHARED_LIBS)
    # Determine the library file name
    if(WIN32)
        set(LIBSVM_SHARED_LIB_NAME "svm.dll")
    elseif(APPLE)
        set(LIBSVM_SHARED_LIB_NAME "libsvm.dylib")
    else()
        set(LIBSVM_SHARED_LIB_NAME "libsvm.so.${PROJECT_VERSION_MAJOR}")
    endif()
    
    # Copy library after build
    add_custom_command(
        TARGET svm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:svm>
            ${CMAKE_CURRENT_SOURCE_DIR}/libsvm/
        COMMENT "Copying libsvm to Python package directory"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

if(Python3_FOUND)
    # Install Python package
    install(DIRECTORY libsvm
        DESTINATION ${CMAKE_INSTALL_DATADIR}/libsvm/python
        FILES_MATCHING
            PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE
    )
    
    install(FILES setup.py MANIFEST.in README
        DESTINATION ${CMAKE_INSTALL_DATADIR}/libsvm/python
    )
endif()

# ============================================================================
# Custom target for pip installation
# ============================================================================

if(Python3_FOUND)
    add_custom_target(python-install
        COMMAND ${Python3_EXECUTABLE} -m pip install -e .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Installing Python bindings with pip"
        DEPENDS svm
    )
endif()
