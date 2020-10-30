//
// Created by dkafri on 10/30/20.
//

#ifndef QSIM_INTERFACE_INCLUDE_PYBIND_INTERFACE_H_
#define QSIM_INTERFACE_INCLUDE_PYBIND_INTERFACE_H_

#include <pybind11/numpy.h>
#include "sampling.h"
/** Interface object for collecting samples
 *
 * Constructor requires num_threads and max_qubits.
 *
 * Attributes:
 * initial_state (private) - KState. Plus function to set it using input array.
 * initial_registers (private) - Setter
 * ops(private) - Function to add COperations and KOperations sequentially.
 * rng(private) - Function to set seed;
 * register_order(private) - setter
 *
 * Other methods:
 * sample_states(num_samples) - assumes initial_state/registers/register_order has already
 * been set. returns a struct storing
 *   a vector of final state vectors (c aligned)
 *   a vector of final qubit_axis (after doing c_align these should all match...)
 *   a 2d array of recorded register values, in register order.
 *
 * */

template<typename Simulator>
class Sampler {
 public:
  using fp_type = typename Simulator::fp_type;
  using StateSpace = typename Simulator::StateSpace;

  size_t max_qubits; /** Sets max required memory for representing state.*/

  /** Basic constructor */
  // We do not allocate memory for the constructed initial state since we
  // assume memory will be externally allocated.
  Sampler(size_t num_threads, size_t max_qubits)
      : num_threads(num_threads),
        max_qubits(max_qubits),
        init_kstate(num_threads, 1, std::vector<std::string>{}) {
    std::random_device rd;
    rgen = std::mt19937(rd());
  };

/** Seed the random number generator */
  void set_random_seed(size_t seed) { rgen = std::mt19937(seed); }

  void set_initial_registers(const RegisterMap& registers) {
    init_registers.clear();
    for (const auto& k_v: registers) init_registers.insert(k_v);
  };

  void set_register_order(const std::vector<std::string>& order) {
    register_order = order;
  };

  /** Set pointer to initial state vector data.
   *
   * User is responsible for allocating a vector of 2 * 2^max_qubits fp_type.
   * */
  void bind_initial_state(pybind11::array_t<typename Simulator::fp_type> array,
                          const std::vector<std::string>& axes) {
    pybind11::buffer_info buffer = array.request();

    size_t expected_size = 2 * (size_t{1} << max_qubits);
    if (buffer.size != expected_size)
      throw std::runtime_error("Input array size does not match max_qubits.");

    auto ptr = static_cast<typename Simulator::fp_type*>(buffer.ptr);
    init_kstate = std::move(KState<Simulator>(num_threads, max_qubits, axes,
                                              ptr));

    init_kstate.print_amplitudes();
  }

 private:
  size_t num_threads; /** Number of multi-threads for simulation.*/

  KState<Simulator> init_kstate; /** Stores initial state vector.*/
  RegisterMap init_registers; /** Initial classical registers */
  std::vector<Operation<fp_type>> ops; /** Operations to sample over.*/
  std::mt19937 rgen; /** random number generator*/
  std::vector<std::string> register_order; /** Order of all saved registers.
 * This is used to determine which virtual registers are recorded and also the
 * output array.*/


};

#endif //QSIM_INTERFACE_INCLUDE_PYBIND_INTERFACE_H_
