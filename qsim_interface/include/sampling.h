//
// Created by dkafri on 10/23/20.
//

#ifndef QSIM_INTERFACE_INCLUDE_SAMPLING_H_
#define QSIM_INTERFACE_INCLUDE_SAMPLING_H_

#include <util.h>
#include "k_ops.h"
#include "state_rep.h"
#include "c_ops.h"
#include "variant.hpp"

using RegisterMap=std::unordered_map<std::string, size_t>;

/** \brief Sample a KOperation on a state.
 *
 * @param op: The KOperation to sample from.
 * @param kstate: The current system state.
 * @param tmp_state: Working memory. Must have the same max_qubit size as
 *     kstate.
 * @param registers: Current classical register table.
 * @param cutoff: Random number between 0 and 1 used for sampling.
 * */
template<typename Simulator, typename fp_type>
static inline void sample_kop(KOperation<fp_type>& op,
                              KState<Simulator>& k_state,
                              KState<Simulator>& tmp_state,
                              RegisterMap& registers,
                              double cutoff) {

  //Extract conditional channel given current registers
  auto channel = op.channel_at(registers);

  //Only need to record a copy if more than one operator may be sampled.
  if (channel.size() > 1)
    tmp_state.copy_from(k_state);
  unsigned k_ind = 0;
  double norm2 = 1.0;

  //Sample operators
  for (auto& k_op : channel) {
    //Add required axes
    for (const auto& ax: k_op.added_axes) {
      k_state.add_qubit(ax);
      tmp_state.add_qubit(ax);
    }

    //Permute and apply matrix
    k_state.permute_and_apply(k_op.matrix, k_op.qubit_axes);

    if (channel.size() > 1) // norm must be 1 if only one operator present.
      norm2 = k_state.norm_squared();
    cutoff -= norm2;
    if (cutoff < 0) { // operator sampled
      // Apply swaps
      for (auto ii = 0; ii < k_op.swap_sources.size(); ii++) {
        k_state.transfer_qubits(k_op.swap_sources[ii], k_op.swap_sinks[ii]);
      }
      // Remove axes
      k_state.remove_qubits_of(k_op.removed_axes);
      //Normalize if needed
      if (channel.size() > 1)
        k_state.rescale(1 / sqrt(norm2));
      break;
    }
    // Operator not sampled -> backtrack. Copy original vector before
    // removing qubits. We need to do this because the removed qubits must be
    // in the zero state before they are removed.
    k_state.copy_from(tmp_state);
    // Qubit removal is free because the added qubits are in the end.
    k_state.remove_qubits_of(k_op.added_axes);
    tmp_state.remove_qubits_of(k_op.added_axes);
    k_ind++;
  }

  if (op.is_recorded) registers[op.label] = k_ind;

}

/** Variant type for classical and quantum operations.*/
template<typename fp_type>
using Operation = mpark::variant<COperation, KOperation<fp_type>>;

/** Apply a classical or quantum sampling operation.*/
template<typename fp_type, typename Simulator>
inline void sample_op(Operation<fp_type>& op,
                      KState<Simulator>& k_state,
                      KState<Simulator>& tmp_state,
                      RegisterMap& registers,
                      double cutoff) {
  if (mpark::holds_alternative<COperation>(op)) {
    const auto& c_op = mpark::get<COperation>(op);
    c_op.apply(registers, cutoff);
#ifdef DEBUG_SAMPLING
    std::cout << "new register values:\n";
    for (const auto& k_v: registers)
      std::cout << k_v.first << ": " << k_v.second << std::endl;
#endif
  } else if (mpark::holds_alternative<KOperation<fp_type>>(op)) {
    auto& k_op = mpark::get<KOperation<fp_type>>(op);
    sample_kop(k_op, k_state, tmp_state, registers, cutoff);
#ifdef DEBUG_SAMPLING
    std::cout << "quantum state after " << k_op.label << std::endl;
    k_state.print_amplitudes();
    std::cout << "qubit axes: ";
    k_state.print_qubit_axes();
#endif
  }
}

/** Whether a classical or quantum operation is virtual.*/
static auto IsVirtual = [](const auto& op) { return op.is_virtual; };

template<typename fp_type>
static inline bool is_virtual(const Operation<fp_type>& current_op) {
  return mpark::visit(IsVirtual, current_op);
}

/** \brief Carry out a sequence of sampling operations on a state.
 *
 * @param ops - Sequence of operations to sample from.
 * @param k_state - Initial state of the system. This will store the final
 * system state at the end of the calculation.
 * @param tmp_state - State used for working memory. Must have the same capacity
 * as k_state.
 * @param init_registers - Initial values assigned to any classical registers.
 * @param rng - Random number generator.
 * @param saved_virtuals - Registers that are created or modified by virtual
 *     operations are only saved if they are in this set. If this set is empty
 *     then virtual operations are skipped altogether.
 * */
template<typename fp_type, typename Simulator>
RegisterMap sample_sequence(std::vector<Operation<fp_type>>& ops,
                            KState<Simulator>& k_state,
                            KState<Simulator>& tmp_state,
                            const RegisterMap& init_registers,
                            std::mt19937& rng,
                            const std::set<std::string>& saved_virtuals) {
  RegisterMap registers(init_registers);
  double cutoff;

  if (!saved_virtuals.empty()) { //implementation with virtual operations
    // Allocated memory for concrete state storage. There may
    // be some overhead from allocating here instead of at a higher level.
    KState<Simulator> last_concrete_state(k_state);
    RegisterMap last_concrete_registers(init_registers);

    std::vector<bool> v_starts(ops.size());
    std::vector<bool> v_stops(ops.size());

    //determine virtual starts and stops
    bool last_virtual = false;
    for (size_t ii = 0; ii < ops.size(); ii++) {
      const Operation<fp_type>& current_op = ops[ii];
      bool is_virt = is_virtual(current_op);
      if (is_virt && !last_virtual) {
        v_starts[ii] = true;
      }
      if (!is_virt && last_virtual) {
        v_stops[ii - 1] = true;
      }
      last_virtual = is_virt;
    }
    //Last operation a virtual stop if it is virtual
    v_stops[ops.size() - 1] = is_virtual(ops[ops.size() - 1]);

    //Sample each operation
    for (size_t ii = 0; ii < ops.size(); ii++) {

      if (v_starts[ii]) { // record last concrete state
        last_concrete_state.copy_from(k_state);
        last_concrete_registers.clear();
        for (const auto& k_v : registers)
          last_concrete_registers.insert(k_v);
      }

      cutoff = qsim::RandomValue(rng, 1.0);
      sample_op(ops[ii], k_state, tmp_state, registers, cutoff);

      if (v_stops[ii]) {//revert to last concrete state
        k_state.copy_from(last_concrete_state);
        //record only saved_virtuals
        for (const auto& k_v: registers) {
          if (saved_virtuals.count(k_v.first))
            last_concrete_registers[k_v.first] = k_v.second;
        }
        registers.clear();
        for (const auto& k_v:last_concrete_registers)
          registers.insert(k_v);
      }

    }

  } else { //implementation without virtual registers

    //Sample each operation. Skip virtual operations.
    for (auto& op: ops) {
      if (mpark::visit(IsVirtual, op))
        continue;
      cutoff = qsim::RandomValue(rng, 1.0);
      sample_op(op, k_state, tmp_state, registers, cutoff);
    }
  }

  return registers;

}

/** Collect many samples for a sequence of operations.
 *
 * inputs:
 * vector of operations (we can create a python function that emplaces
 * KOperation or COperation into this vector. How do we expose this vector?)
 * c aligned initial state vector
 * initial axis labels (qubit_axis)
 * initial registers
 * register order (for all saved final registers. should contain measurements,
 * added registers... can we search this to determine saved_virtuals?)
 * random seed
 * num_samples
 *
 * outputs:
 * a struct storing a vector of final state vectors (c aligned)
 * a vector of final qubit_axis (after doing c_align these should all match...)
 * a 2d array of recorded register values, in register order.
 *
 * */



#endif //QSIM_INTERFACE_INCLUDE_SAMPLING_H_
