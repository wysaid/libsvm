# ============================================================================
# LibSVM Command-Line Tools
# ============================================================================

# Common include directory
include_directories(${CMAKE_SOURCE_DIR}/src)

# ============================================================================
# svm-train
# ============================================================================

add_executable(svm-train svm-train.c)
target_link_libraries(svm-train PRIVATE svm)

if(UNIX)
    target_link_libraries(svm-train PRIVATE m)
endif()

# ============================================================================
# svm-predict
# ============================================================================

add_executable(svm-predict svm-predict.c)
target_link_libraries(svm-predict PRIVATE svm)

if(UNIX)
    target_link_libraries(svm-predict PRIVATE m)
endif()

# ============================================================================
# svm-scale
# ============================================================================

add_executable(svm-scale svm-scale.c)

# svm-scale doesn't need libsvm, it's a standalone tool

# ============================================================================
# Installation
# ============================================================================

install(TARGETS svm-train svm-predict svm-scale
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
