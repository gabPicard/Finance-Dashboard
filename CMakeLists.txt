cmake_minimum_required(VERSION 3.15)
project(finance_dashboard_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisation flags
if(NOT MSVC)
    add_compile_options(-O3 -march=native)
else()
    add_compile_options(/O2)
endif()

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Find Eigen3 (header-only)
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    # Fallback: try pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(EIGEN3 QUIET eigen3)
    endif()
    if(NOT EIGEN3_FOUND)
        # Last resort: download via FetchContent
        include(FetchContent)
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG        3.4.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(eigen)
        set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
    endif()
endif()

# Collect source files
set(CORE_SOURCES
    src/core/optimizer.cpp
    src/core/backtest_engine.cpp
    src/core/spread_calc.cpp
    src/core/bindings.cpp
)

# Build the pybind11 extension module
pybind11_add_module(_core ${CORE_SOURCES})

# Include Eigen headers
if(TARGET Eigen3::Eigen)
    target_link_libraries(_core PRIVATE Eigen3::Eigen)
elseif(DEFINED EIGEN3_INCLUDE_DIR)
    target_include_directories(_core PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# Install the module at the package root so `import _core` works
install(TARGETS _core DESTINATION .)
