cmake_minimum_required(VERSION 3.11)
project(qsim)

set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp")

include_directories(../lib include)


add_executable(k_state kstate_example.cc)
add_executable(tests interface_tests.cc)
add_executable(qop_sampling qop_sampling.cc)
add_executable(simulator_example simulator_example.cc)