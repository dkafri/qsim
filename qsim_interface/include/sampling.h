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
#include <test_utils.h>

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

#ifndef NDEBUG
  double norm2_tot = 0.0;
  double original_cutoff = cutoff;
  double original_norm2 = k_state.norm_squared();
#endif

  //Sample operators
  for (auto& k_op : channel) {
    //Add required qubits to axes
    for (const auto& ax: k_op.added_axes) {
      k_state.add_qubit(ax);
      tmp_state.add_qubit(ax);
    }

    //Permute and apply matrix
    k_state.permute_and_apply(k_op.matrix, k_op.qubit_axes);


    if (channel.size() > 1) // norm must be 1 if only one operator present.
      norm2 = k_state.norm_squared();
#ifndef NDEBUG
    norm2_tot += norm2;
#endif
    cutoff -= norm2;
    if (cutoff < 0) { // operator sampled

      // Apply swaps
      for (std::size_t ii = 0; ii < k_op.swap_sources.size(); ii++) {
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

    // Rarely floating point errors prevent the total Kraus operator
    // probabilities from summing to one. We correct this case below, but want
    // to confirm that the total cutoff is only very slightly positive.
    ASSERT(k_ind < channel.size() || cutoff < 5e-7,
           "\nNo kraus operator sampled for KOperation labeled ("
               << op.label << ").\n Remaining cutoff: " << cutoff
               << ",\nmost recent norm2: " << norm2
               << ",\ntotal summed norm2-1: " << norm2_tot - 1
               << ",\noriginal cutoff-1: " << original_cutoff - 1
               << ",\ntotal summed - original = "
               << norm2_tot - original_cutoff
               << ",\noriginal state vector norm -1: " << original_norm2 - 1
               << ".\n");

  }
  // TODO: Fix this if qsim norm is revised.
  k_ind = (k_ind == channel.size()) ? k_ind - 1 : k_ind;

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
  } else if (mpark::holds_alternative<KOperation<fp_type>>(op)) {
    auto& k_op = mpark::get<KOperation<fp_type>>(op);
    sample_kop(k_op, k_state, tmp_state, registers, cutoff);
  }
}

/** Whether a classical or quantum operation is virtual.*/
static auto IsVirtual = [](const auto& op) { return op.is_virtual; };

template<typename fp_type>
inline bool is_virtual(const Operation<fp_type>& op) {
  return mpark::visit(IsVirtual, op);
}

template<typename fp_type>
struct SavedVirtualsVisitor {
  using RegisterSet = std::set<std::string>;

  RegisterSet operator()(const KOperation<fp_type>& k_op) const {
    if (k_op.is_recorded && k_op.is_virtual)
      return RegisterSet({k_op.label});
    else
      return RegisterSet{};
  }

  RegisterSet operator()(const COperation& c_op) const {
    // Basically we say every virtually generated register should be
    // tracked in the course of a simulation. Note that this can lead to bugs
    // if the same virtual register is created more than added more than once.
    if (c_op.is_virtual)
      return c_op.added;
    else
      return RegisterSet{};
  }
};

/** All registers to be tracked during a simulation that are generated by a
 *  virtual operation. These are not necessarily recorded at the end of
 *  the sampling (that is set by register order).*/
template<typename fp_type>
std::set<std::string> saved_virtuals(const Operation<fp_type>& op) {
  return mpark::visit(SavedVirtualsVisitor<fp_type>(), op);
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
      if (is_virtual(op))
        continue;
      cutoff = qsim::RandomValue(rng, 1.0);
      sample_op(op, k_state, tmp_state, registers, cutoff);
    }
  }

  return registers;

}

#endif //QSIM_INTERFACE_INCLUDE_SAMPLING_H_
