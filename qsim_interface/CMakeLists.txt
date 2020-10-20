cmake_minimum_required(VERSION 3.11)
project(qsim)

set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp")

include_directories(../lib include)


add_executable(simple_example simple_example.cc)
add_executable(destroy_qubit destroy_qubit.cc)
add_executable(k_state kstate_example.cc)
add_executable(tests interface_tests.cc)
add_executable(gate_order qubit_order_gate.cc)
add_executable(qop_sampling qop_sampling.cc)