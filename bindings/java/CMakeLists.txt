# ============================================================================
# LibSVM Java Bindings
# ============================================================================
#
# The Java implementation of LibSVM is a pure Java port and does NOT depend
# on the C library. It provides the same functionality but runs entirely
# in the JVM.
#
# Build process:
#   1. Generate svm.java from svm.m4 using m4 preprocessor
#   2. Compile all Java files
#   3. Package into libsvm.jar
#
# Usage:
#   cmake -DLIBSVM_BUILD_JAVA=ON ..
#   cmake --build . --target java-bindings
#
# ============================================================================

message(STATUS "Configuring Java bindings")

# ============================================================================
# Find required tools
# ============================================================================

find_package(Java COMPONENTS Development)
find_program(M4_EXECUTABLE m4)

if(NOT Java_FOUND)
    message(WARNING "Java Development Kit not found - Java bindings will not be built")
    return()
endif()

if(NOT M4_EXECUTABLE)
    message(WARNING "m4 preprocessor not found - Java bindings will not be built")
    return()
endif()

include(UseJava)

# ============================================================================
# Java source files
# ============================================================================

set(JAVA_LIBSVM_SOURCES
    libsvm/svm_model.java
    libsvm/svm_node.java
    libsvm/svm_parameter.java
    libsvm/svm_print_interface.java
    libsvm/svm_problem.java
)

set(JAVA_APP_SOURCES
    svm_train.java
    svm_predict.java
    svm_scale.java
    svm_toy.java
)

# ============================================================================
# Generate svm.java from svm.m4
# ============================================================================

set(SVM_JAVA_M4 ${CMAKE_CURRENT_SOURCE_DIR}/libsvm/svm.m4)
set(SVM_JAVA_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/libsvm/svm.java)

add_custom_command(
    OUTPUT ${SVM_JAVA_GENERATED}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/libsvm
    COMMAND ${M4_EXECUTABLE} ${SVM_JAVA_M4} -o ${SVM_JAVA_GENERATED}
    DEPENDS ${SVM_JAVA_M4}
    COMMENT "Generating svm.java from svm.m4"
)

add_custom_target(generate-svm-java
    DEPENDS ${SVM_JAVA_GENERATED}
)

# ============================================================================
# Build JAR file
# ============================================================================

# Copy source files to build directory for compilation
set(JAVA_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/java-build)
file(MAKE_DIRECTORY ${JAVA_BUILD_DIR})
file(MAKE_DIRECTORY ${JAVA_BUILD_DIR}/libsvm)

# List Java source files for dependency tracking
set(JAVA_LIBSVM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/libsvm/svm_model.java
    ${CMAKE_CURRENT_SOURCE_DIR}/libsvm/svm_node.java
    ${CMAKE_CURRENT_SOURCE_DIR}/libsvm/svm_parameter.java
    ${CMAKE_CURRENT_SOURCE_DIR}/libsvm/svm_print_interface.java
    ${CMAKE_CURRENT_SOURCE_DIR}/libsvm/svm_problem.java
)

set(JAVA_APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/svm_train.java
    ${CMAKE_CURRENT_SOURCE_DIR}/svm_predict.java
    ${CMAKE_CURRENT_SOURCE_DIR}/svm_scale.java
    ${CMAKE_CURRENT_SOURCE_DIR}/svm_toy.java
)

# Custom target to compile Java and create JAR
add_custom_target(java-bindings ALL
    DEPENDS generate-svm-java
            ${JAVA_LIBSVM_SOURCES}
            ${JAVA_APP_SOURCES}
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        ${CMAKE_CURRENT_SOURCE_DIR}/libsvm 
        ${JAVA_BUILD_DIR}/libsvm
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SVM_JAVA_GENERATED}
        ${JAVA_BUILD_DIR}/libsvm/svm.java
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/svm_train.java
        ${CMAKE_CURRENT_SOURCE_DIR}/svm_predict.java
        ${CMAKE_CURRENT_SOURCE_DIR}/svm_scale.java
        ${CMAKE_CURRENT_SOURCE_DIR}/svm_toy.java
        ${JAVA_BUILD_DIR}/
    COMMAND ${Java_JAVAC_EXECUTABLE} -d ${JAVA_BUILD_DIR}
        --release 11
        ${JAVA_BUILD_DIR}/libsvm/*.java
        ${JAVA_BUILD_DIR}/*.java
    COMMAND ${Java_JAR_EXECUTABLE} cvf ${CMAKE_CURRENT_BINARY_DIR}/libsvm.jar
        -C ${JAVA_BUILD_DIR} .
    WORKING_DIRECTORY ${JAVA_BUILD_DIR}
    COMMENT "Building Java bindings and creating libsvm.jar"
)

# ============================================================================
# Installation
# ============================================================================

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libsvm.jar
    DESTINATION ${CMAKE_INSTALL_DATADIR}/libsvm/java
)

install(DIRECTORY libsvm
    DESTINATION ${CMAKE_INSTALL_DATADIR}/libsvm/java
    FILES_MATCHING
        PATTERN "*.java"
        PATTERN "*.m4"
)

install(FILES
    svm_train.java
    svm_predict.java
    svm_scale.java
    svm_toy.java
    DESTINATION ${CMAKE_INSTALL_DATADIR}/libsvm/java
)
