cmake_minimum_required(VERSION 3.11)
project(qsim)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp")
message(STATUS "compiler flags ${CMAKE_CXX_FLAGS}")

find_package(PythonInterp REQUIRED)
message(STATUS "${PYTHON_EXECUTABLE}")

message(STATUS "C++ compiler : ${CMAKE_CXX_COMPILER}")

include_directories(../lib include)


add_executable(state_rep_tests state_rep_tests.cc)
add_executable(sampling_tests sampling_tests.cc)
add_executable(kop_sampling kop_sampling.cc)
add_executable(kop_sampling_sequence kop_sampling_sequence.cc)

## Include pybind11
# To be able to use pybind, we also need to add it as a "submodule" as defined
# here:
# https://pybind11.readthedocs.io/en/stable/installing.html
include(FetchContent)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.2.4
)
FetchContent_GetProperties(pybind11)
if (NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif ()

pybind11_add_module(pybind_interface pybind_interface.cc)
pybind11_add_module(pybind_example pybind_example.cc)
