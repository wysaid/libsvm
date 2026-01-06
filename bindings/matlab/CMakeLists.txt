# ============================================================================
# LibSVM MATLAB/MEX Bindings
# ============================================================================
#
# This builds MEX files for MATLAB/Octave that link against the LibSVM
# C library. The MEX files provide MATLAB functions:
#   - svmtrain: Train an SVM model
#   - svmpredict: Predict using a trained model
#   - libsvmread: Read data in LIBSVM format
#   - libsvmwrite: Write data in LIBSVM format
#
# Usage:
#   cmake -DLIBSVM_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/path/to/matlab ..
#   cmake --build . --target matlab-bindings
#
# For Octave:
#   Use the make.m script in Octave instead
#
# ============================================================================

message(STATUS "Configuring MATLAB bindings")

# ============================================================================
# Find MATLAB
# ============================================================================

find_package(Matlab COMPONENTS MX_LIBRARY MEX_COMPILER)

if(NOT Matlab_FOUND)
    message(WARNING "MATLAB not found - MATLAB bindings will not be built. "
                    "Set Matlab_ROOT_DIR to your MATLAB installation path, or "
                    "use the make.m script in Octave.")
    return()
endif()

message(STATUS "MATLAB found: ${Matlab_ROOT_DIR}")
message(STATUS "MEX extension: ${Matlab_MEX_EXTENSION}")

# ============================================================================
# MEX Source Files
# ============================================================================

set(MATLAB_COMMON_SOURCES
    svm_model_matlab.c
    svm_model_matlab.h
)

# ============================================================================
# svmtrain MEX
# ============================================================================

matlab_add_mex(
    NAME svmtrain_mex
    OUTPUT_NAME svmtrain
    SRC svmtrain.c ${MATLAB_COMMON_SOURCES}
    LINK_TO svm
)

target_include_directories(svmtrain_mex PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# ============================================================================
# svmpredict MEX
# ============================================================================

matlab_add_mex(
    NAME svmpredict_mex
    OUTPUT_NAME svmpredict
    SRC svmpredict.c ${MATLAB_COMMON_SOURCES}
    LINK_TO svm
)

target_include_directories(svmpredict_mex PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# ============================================================================
# libsvmread MEX
# ============================================================================

matlab_add_mex(
    NAME libsvmread_mex
    OUTPUT_NAME libsvmread
    SRC libsvmread.c
)

# ============================================================================
# libsvmwrite MEX
# ============================================================================

matlab_add_mex(
    NAME libsvmwrite_mex
    OUTPUT_NAME libsvmwrite
    SRC libsvmwrite.c
)

# ============================================================================
# Combined target
# ============================================================================

add_custom_target(matlab-bindings
    DEPENDS svmtrain_mex svmpredict_mex libsvmread_mex libsvmwrite_mex
    COMMENT "Building MATLAB MEX bindings"
)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS svmtrain_mex svmpredict_mex libsvmread_mex libsvmwrite_mex
    LIBRARY DESTINATION ${CMAKE_INSTALL_DATADIR}/libsvm/matlab
    RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/libsvm/matlab
)

install(FILES README make.m
    DESTINATION ${CMAKE_INSTALL_DATADIR}/libsvm/matlab
)
