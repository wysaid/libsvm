# ============================================================================
# LibSVM Core Library
# ============================================================================

set(LIBSVM_SOURCES
    svm.cpp
)

set(LIBSVM_HEADERS
    svm.h
)

# ============================================================================
# Library Target
# ============================================================================

add_library(svm ${LIBSVM_SOURCES} ${LIBSVM_HEADERS})
add_library(LibSVM::svm ALIAS svm)

target_include_directories(svm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Set library version
set_target_properties(svm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER svm.h
)

# Windows DLL export definitions
if(WIN32 AND BUILD_SHARED_LIBS)
    target_sources(svm PRIVATE svm.def)
endif()

# ============================================================================
# OpenMP Support
# ============================================================================

if(LIBSVM_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(svm PRIVATE OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Math Library (required on some platforms)
# ============================================================================

if(UNIX AND NOT APPLE)
    target_link_libraries(svm PUBLIC m)
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS svm
    EXPORT LibSVMTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
