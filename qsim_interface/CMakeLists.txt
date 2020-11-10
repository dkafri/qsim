cmake_minimum_required(VERSION 3.11)
project(qsim)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp -DNDEBUG")
#set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp -DDEBUG_SAMPLING")


find_package(PythonInterp REQUIRED)
message(STATUS "${PYTHON_EXECUTABLE}")
message(STATUS "compiler flags ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ compiler : ${CMAKE_CXX_COMPILER}")

include_directories(../lib include)


add_executable(state_rep_tests state_rep_tests.cc)
add_executable(sampling_tests sampling_tests.cc)

add_subdirectory(extern/pybind11)

pybind11_add_module(qsim_kraus_sim qsim_kraus_sim.cc)
