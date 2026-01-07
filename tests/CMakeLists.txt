# ============================================================================
# LibSVM Test Suite
# ============================================================================

include(FetchContent)

# ============================================================================
# Google Test Framework
# ============================================================================

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# ============================================================================
# Test Data
# ============================================================================

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")

# Copy test data to build directory
file(COPY ${CMAKE_SOURCE_DIR}/examples/data/heart_scale
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

# ============================================================================
# Unit Tests
# ============================================================================

add_executable(unit_tests
    unit/test_svm_node.cpp
    unit/test_svm_parameter.cpp
    unit/test_svm_problem.cpp
    unit/test_kernel.cpp
    unit/test_model.cpp
    common/test_utils.cpp
)

target_include_directories(unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(unit_tests PRIVATE
    svm
    GTest::gtest
    GTest::gtest_main
)

target_compile_definitions(unit_tests PRIVATE
    TEST_DATA_DIR="${TEST_DATA_DIR}"
)

gtest_discover_tests(unit_tests)

# ============================================================================
# Integration Tests
# ============================================================================

add_executable(integration_tests
    integration/test_train_predict.cpp
    integration/test_cross_validation.cpp
    integration/test_model_io.cpp
    integration/test_probability.cpp
    common/test_utils.cpp
)

target_include_directories(integration_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(integration_tests PRIVATE
    svm
    GTest::gtest
    GTest::gtest_main
)

target_compile_definitions(integration_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_BINARY_DIR}/data"
)

gtest_discover_tests(integration_tests)

# ============================================================================
# Memory Tests (with AddressSanitizer/Valgrind support)
# ============================================================================

add_executable(memory_tests
    memory/test_memory_leaks.cpp
    memory/test_resource_management.cpp
    common/test_utils.cpp
)

target_include_directories(memory_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(memory_tests PRIVATE
    svm
    GTest::gtest
    GTest::gtest_main
)

target_compile_definitions(memory_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_BINARY_DIR}/data"
)

# Enable AddressSanitizer for memory tests if available
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(LIBSVM_ENABLE_ASAN)
        target_compile_options(memory_tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(memory_tests PRIVATE -fsanitize=address)
    endif()
endif()

gtest_discover_tests(memory_tests)

# ============================================================================
# Upstream Comparison Tests
# ============================================================================

option(LIBSVM_BUILD_UPSTREAM_COMPARISON "Build upstream comparison tests" OFF)

if(LIBSVM_BUILD_UPSTREAM_COMPARISON)
    # Check if upstream subtree exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/upstream_libsvm/svm.cpp")
        message(STATUS "Building upstream comparison tests")
        
        # Build upstream library as a separate target
        add_library(svm_upstream STATIC
            upstream_libsvm/svm.cpp
        )
        target_include_directories(svm_upstream PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/upstream_libsvm
        )
        
        add_executable(upstream_comparison_tests
            comparison/test_upstream_comparison.cpp
            common/test_utils.cpp
        )
        
        target_include_directories(upstream_comparison_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/common
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/upstream_libsvm
        )
        
        target_link_libraries(upstream_comparison_tests PRIVATE
            svm
            svm_upstream
            GTest::gtest
            GTest::gtest_main
        )
        
        target_compile_definitions(upstream_comparison_tests PRIVATE
            TEST_DATA_DIR="${CMAKE_CURRENT_BINARY_DIR}/data"
            UPSTREAM_AVAILABLE
        )
        
        gtest_discover_tests(upstream_comparison_tests)
    else()
        message(WARNING "Upstream libsvm not found. Run: git read-tree --prefix=tests/upstream_libsvm -u upstream")
    endif()
endif()

# ============================================================================
# OpenCV Comparison Tests
# ============================================================================

option(LIBSVM_BUILD_OPENCV_COMPARISON "Build OpenCV SVM comparison tests" OFF)

if(LIBSVM_BUILD_OPENCV_COMPARISON)
    find_package(OpenCV QUIET COMPONENTS core ml)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
        message(STATUS "Building OpenCV SVM comparison tests")
        
        add_executable(opencv_comparison_tests
            comparison/test_opencv_comparison.cpp
            common/test_utils.cpp
        )
        
        target_include_directories(opencv_comparison_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/common
            ${CMAKE_SOURCE_DIR}/src
        )
        
        target_link_libraries(opencv_comparison_tests PRIVATE
            svm
            ${OpenCV_LIBS}
            GTest::gtest
            GTest::gtest_main
        )
        
        target_compile_definitions(opencv_comparison_tests PRIVATE
            TEST_DATA_DIR="${CMAKE_CURRENT_BINARY_DIR}/data"
            OPENCV_AVAILABLE
        )
        
        gtest_discover_tests(opencv_comparison_tests)
    else()
        message(WARNING "OpenCV not found. OpenCV comparison tests will not be built.")
    endif()
endif()

# ============================================================================
# Custom Test Target
# ============================================================================

add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS unit_tests integration_tests memory_tests
    COMMENT "Running all LibSVM tests"
)
