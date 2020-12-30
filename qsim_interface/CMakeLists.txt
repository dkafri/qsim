cmake_minimum_required(VERSION 3.11)
project(qsim)

# To do a "debug" compilation, add the following flag to cmake:
# -DCMAKE_BUILD_TYPE=Debug
# This generates some tests and uses the ggdb3 compilation flag for easier
# memory leak tracking with valgrind. It will also considerably slow down the
# compilation.

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Change CXX flags depending on build type.
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${default_build_type}' as none was
    specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
    message(STATUS "Compiling release build.")
    set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp -DNDEBUG")
else ()
    message(STATUS "Compiling ${CMAKE_BUILD_TYPE} build.")
    set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp -ggdb3 -Wall")
endif ()
#set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp -DDEBUG_SAMPLING")


find_package(PythonInterp REQUIRED)
message(STATUS "${PYTHON_EXECUTABLE}")
message(STATUS "compiler flags ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ compiler : ${CMAKE_CXX_COMPILER}")

include_directories(../lib include)

if (${CMAKE_BUILD_TYPE} MATCHES "Debug") # also compile tests if in debug mode
    add_executable(state_rep_tests state_rep_tests.cc)
    add_executable(sampling_tests sampling_tests.cc)
endif ()

add_subdirectory(extern/pybind11)

pybind11_add_module(qsim_kraus_sim qsim_kraus_sim.cc)
