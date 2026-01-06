# ============================================================================
# LibSVM Language Bindings
# ============================================================================

# Python bindings
if(LIBSVM_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Java bindings
if(LIBSVM_BUILD_JAVA)
    add_subdirectory(java)
endif()

# MATLAB bindings
if(LIBSVM_BUILD_MATLAB)
    add_subdirectory(matlab)
endif()
