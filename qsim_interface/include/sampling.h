//
// Created by dkafri on 10/23/20.
//

#ifndef QSIM_INTERFACE_INCLUDE_SAMPLING_H_
#define QSIM_INTERFACE_INCLUDE_SAMPLING_H_

#include "k_ops.h"
#include "state_rep.h"

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
inline void sample_op(KOperation<fp_type>& op,
                      KState<Simulator>& k_state,
                      KState<Simulator>& tmp_state,
                      std::unordered_map<std::string, size_t>& registers,
                      double cutoff) {

  //Extract conditional channel given current registers
  auto channel = op.channel_at(registers);

  //Only need to record a copy if more than one operator may be sampled.
  if (channel.size() > 1)
    tmp_state.copy_from(k_state);
  unsigned k_ind = 0;
  double norm2;

  //Sample operators
  for (auto& k_op : channel) {
    //Add required axes
    for (const auto& ax: k_op.added_axes) {
      k_state.add_qubit(ax);
      tmp_state.add_qubit(ax);
    }

    //Permute and apply matrix
    k_state.permute_and_apply(k_op.matrix, k_op.qubit_axes);

    norm2 = k_state.norm_squared();
    cutoff -= norm2;
    if (cutoff < 0) { // operator sampled
      // Apply swaps
      for (auto ii = 0; ii < k_op.swap_sources.size(); ii++) {
        k_state.transfer_qubits(k_op.swap_sources[ii], k_op.swap_sinks[ii]);
      }
      // Remove axes
      k_state.remove_qubits_of(k_op.removed_axes);
      //Normalize
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

  //Update registers if operation is a measurement
  if (op.is_measurement) registers[op.label] = k_ind;

};

#endif //QSIM_INTERFACE_INCLUDE_SAMPLING_H_
