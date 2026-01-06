# ============================================================================
# svm-toy - Qt GUI Example for LibSVM
# ============================================================================

# Check for QTDIR environment variable
if(DEFINED ENV{QTDIR})
    message(STATUS "QTDIR environment variable found: $ENV{QTDIR}")
    set(CMAKE_PREFIX_PATH "$ENV{QTDIR}" ${CMAKE_PREFIX_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{QTDIR}/lib/cmake")
endif()

# Try to find Qt6 first, then fall back to Qt5
find_package(Qt6 COMPONENTS Widgets Gui Core QUIET)

if(Qt6_FOUND)
    message(STATUS "Found Qt6 - building svm-toy with Qt6")
    set(QT_LIBRARIES Qt6::Widgets Qt6::Gui Qt6::Core)
else()
    find_package(Qt5 COMPONENTS Widgets Gui Core QUIET)
    if(Qt5_FOUND)
        message(STATUS "Found Qt5 - building svm-toy with Qt5")
        set(QT_LIBRARIES Qt5::Widgets Qt5::Gui Qt5::Core)
    else()
        message(STATUS "Qt5/Qt6 not found - skipping svm-toy build")
        return()
    endif()
endif()

# Enable Qt MOC (Meta-Object Compiler)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ============================================================================
# svm-toy executable
# ============================================================================

add_executable(svm-toy svm-toy.cpp)

target_include_directories(svm-toy PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(svm-toy
    PRIVATE
        svm
        ${QT_LIBRARIES}
)

target_compile_features(svm-toy PRIVATE cxx_std_11)

# Suppress common warnings for this example
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(svm-toy PRIVATE
        -Wno-sign-conversion
        -Wno-implicit-int-conversion
        -Wno-shorten-64-to-32
        -Wno-deprecated-declarations
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS svm-toy
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
