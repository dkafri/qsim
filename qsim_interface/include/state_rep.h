//
// Implementation of state representation.
// Created by dkafri on 10/12/20.
//



#ifndef PROTOTYPE_STATE_REP_H_
#define PROTOTYPE_STATE_REP_H_

#include <complex>
#include <iostream>
#include <unordered_map>
#include <list>
#include <cassert>
#include <set>
#include <unordered_set>
#include "../../lib/gates_cirq.h"
#include "q_ops.h"

/* Data representation of a state vector with variable axis dimensions.
 *
 * This class allows for tracking and update of state tensor axis labels,
 * including creation and destruction of axes. Although one state array is ever
 * allocated, this modification is handled by performing swaps on qubits
 * (pushing them to the back or retrieving them) that are created or destroyed.
 * Destroying a qubit is equivalent to pushing that qubit to the back (the
 * index which is updated slowest). Subsequent operations then only act on the
 * part of the array data we care about.
 * */
template<typename Simulator>
class KState {
 public:
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename Simulator::fp_type;

  // Methods
  KState() = delete;

  template<typename StringIterable>
  KState(unsigned num_threads,
         unsigned max_qubits,
         const StringIterable axis_labels
  )
      : state_vec(StateSpace(num_threads).Create(max_qubits)),
        num_threads(num_threads), max_qubits(max_qubits) {

    // Assign qubits to axis labels. Assume all axes are initially allocated one
    // qubit.
    for (auto axis_ptr = axis_labels.begin(); axis_ptr != axis_labels.end();
         ++axis_ptr) {
      add_qubit(*axis_ptr);
    }

    // Initialize state vector as zero state.
    StateSpace(num_threads).SetStateZero(state_vec);

  }

  /** Copy constructor*/
  KState(const KState& k_state) :
      state_vec(StateSpace(k_state.num_threads).Create(k_state.max_qubits)),
      num_threads(k_state.num_threads), max_qubits(k_state.max_qubits) {

    //copy axis_qubits and qubit_axis
    for (const auto& axis : k_state.qubit_axis) {
      qubit_axis.push_back(axis);
    }
    for (const auto& key_value : k_state.axis_qubits) {
      axis_qubits[key_value.first] = key_value.second;
    }

    StateSpace(num_threads).Copy(k_state.state_vec, state_vec);

  }

  /** \brief Copy another state into this one.
   *
   * @param source: KState to copy. Must have the same number of threads and
   *    max_qubits.
   * */
  void copy_from(KState& source) {
    assert(max_qubits == source.max_qubits);
    assert(num_threads == source.num_threads);

    // Need to clear the state vector if it is larger than source's

    if (num_active_qubits() > source.num_active_qubits()) {
      State my_state = active_state();
      active_state_space().SetAllZeros(my_state);
    }

    //copy axis_qubits and qubit_axis
    axis_qubits.clear();
    qubit_axis.clear();
    for (const auto& axis : source.qubit_axis) {
      qubit_axis.push_back(axis);
    }
    for (const auto& key_value : source.axis_qubits) {
      axis_qubits[key_value.first] = key_value.second;
    }
    assert(num_active_qubits() == source.num_active_qubits());

    //copy state vector
    State source_state = source.active_state();
    State my_state = active_state();
    StateSpace(num_threads).Copy(source_state, my_state);
  }

  /** State space appropriate to current vector size.*/
  StateSpace active_state_space() const {
    return StateSpace(num_threads);
  }

  /** Simulator appropriate to current vector size.*/
  Simulator active_simulator() const {
    return Simulator(num_threads);
  }

  /* State with current vector size.*/
  State active_state() {
    return StateSpace(num_threads).Create(state_vec.get(), num_active_qubits());
  }

  /** Allocate a qubit to an axis.
   *
   * If the axis does not exist, it is created first. It is not guaranteed that
   * the specified qubit is exactly the zero state (i.e. no entanglement).
   *
   * @param axis: The axis to which to add the qubit to.
   * */
  void add_qubit(const std::string& axis) {
    assert(num_active_qubits() < max_qubits);

    axis_qubits[axis].push_back(num_active_qubits());
    qubit_axis.push_back(axis);

  }

  /** Deallocate a qubit from an axis.
  *
  * The axis must have at least one qubit assigned to it. This operation removes
  * one of the qubits assigned to an axis (the one added most recently). When
  * removing a qubit from an axis, the removed qubit should correspond to
   * exactly the zero state (no entanglement).
  *
  * @param axis: The axis to which to remove a qubit from.
  * */
  void remove_qubit(const std::string& axis) {
    assert(num_active_qubits() > 0);
    assert(!axis_qubits[axis].empty());

    // Swap the last qubit of this axis with the last active qubit.
    const unsigned last_active_q = num_active_qubits() - 1;
    const unsigned removed_q = axis_qubits[axis].back();

    if (last_active_q != removed_q) {
      auto state = active_state();
      std::vector<unsigned> qubits{last_active_q, removed_q};
      active_simulator().ApplyGate(qubits, swap_matrix.data(), state);

      // Update the axis qubit registry for the axis involved in the swap.
      std::string swapped_axis = qubit_axis[last_active_q];
      assert(axis_qubits[swapped_axis].back() == last_active_q);
      axis_qubits[swapped_axis].pop_back();
      axis_qubits[swapped_axis].push_back(removed_q);
      qubit_axis[removed_q] = swapped_axis;
    }
    // Deallocate the removed qubit.
    axis_qubits[axis].pop_back();
    qubit_axis.pop_back();

  }

  /** Remove qubits from a sequence of axes.
   *
   * Removal order is chosen to minimize any required swaps.
   *
   * @param axes: Axes for which all qubits are to be removed. Repeats are
   *     ignored since all qubits are removed.
   * */
  void remove_qubits_of(const std::vector<std::string>& axes) {
    // Removal order is chosen to minimize any required swaps. This is done
    // by ensuring that calls to remove_qubit are invoked for the axis whose
    // most recently added qubit is highest in the order.
    using QubitAndAxis =std::pair<unsigned, std::string>;
    std::list<QubitAndAxis> q_and_ax;

    std::unordered_set<std::string> seen;
    for (const auto& ax : axes) {
      if (seen.find(ax) != seen.end())
        continue; //Skip repeats
      for (const auto& q : qubits_of(ax)) {
        q_and_ax.push_back({q, ax});
      }
    }

    auto cmp = [this](const QubitAndAxis& a, const QubitAndAxis& b) {
      if (a.second == b.second) {
        // If both qubits belong to the same axis, they must be removed
        // in the order in which they are added.
        for (const auto& q : qubits_of(a.second)) {
          if (q == a.first)
            return false;
          if (q == b.first)
            return true;
        }
      }

      return a.first > b.first;
    };
    q_and_ax.sort(cmp);

    for (const auto& pair: q_and_ax) {
      remove_qubit(pair.second);
    }

  }

  /** Add qubits required for a QOperator.*/
  template<size_t num_qubit>
  void add_qubits_for(const QOperator<fp_type, num_qubit>& q_op) {
    for (const auto& ax :q_op.added_axes)
      add_qubit(ax);
  }

  /** Move qubits from one axis to another.
   *
   * This is equivalent to relabeling an axis. The ordering of qubit
   * assignments is preserved.
   *
   * @param src: The axis whose qubits should be transferred.
   * @param dest: The axis receiving qubits. This axis must not have any
   *              allocated qubits.
   * */
  void transfer_qubits(const std::string& src, const std::string& dest) {

    assert(axis_qubits[dest].size() == 0);

    //reassign qubits from source to destination
    axis_qubits[dest] = std::move(axis_qubits[src]);
    axis_qubits[src].clear();

    //Update qubit_axes
    for (const auto& q:axis_qubits[dest])
      qubit_axis[q] = dest;
  }

  /** Print amplitudes of all active qubits to cout.*/
  void print_amplitudes() {

    auto state = active_state();

    uint64_t last = powl(2, num_active_qubits());
    for (size_t ii = 0; ii < last - 1; ii++) {
      std::cout << StateSpace::GetAmpl(state, ii) << ", ";
    }
    std::cout << StateSpace::GetAmpl(state, last - 1) << std::endl;

  }

  /** Rescale the state vector by a constant.*/
  void rescale(double scale) {
    auto state = active_state();
    active_state_space().Multiply(scale, state);
  }

  /** Qubits allocated to a given axis.*/
  std::vector<unsigned> qubits_of(const std::string& axis) {
    const auto& qubits = axis_qubits[axis];
    return std::vector<unsigned>(qubits.begin(), qubits.end());
  }

  /** Qubits allocated to one or more axes */
  std::vector<unsigned> qubits_of(const std::vector<std::string>& axes) {
    std::vector<unsigned> out{};

    for (const auto& axis : axes)
      out.insert(out.end(), axis_qubits[axis].begin(), axis_qubits[axis].end());

    return out;
  }

  /** Apply a matrix to the specified axes.
   *
   * @param matrix: Square matrix (array of array, complex float type) to apply
   *     to the state.
   * @param axes: Order of axes corresponding to the qubits of the matrix.
   *     Must satisfy matrix.size() == 2^axes.size()
   * */
  void apply(const qsim::Cirq::Matrix2q<fp_type>& matrix,
             const std::vector<std::string>& axes) {


    // Create appropriate gate structure matching qubit ordering.
    auto qubits = qubits_of(axes);
    assert(qubits.size() == 2);
    auto gate = qsim::Cirq::MatrixGate2<fp_type>::Create(0,
                                                         qubits[1],
                                                         qubits[0],
                                                         matrix);
    // Apply gate
    auto state = active_state();
    active_simulator().ApplyGate(gate.qubits, gate.matrix.data(), state);
  }

  /** Apply a matrix to the specified axes.
 *
 * @param matrix: Square matrix (array of array, complex float type) to apply
 *     to the state.
 * @param axes: Order of axes corresponding to the qubits of the matrix.
 *     Must satisfy matrix.size() == 2^axes.size()
 * */
  void apply(const qsim::Cirq::Matrix1q<fp_type>& matrix,
             const std::vector<std::string>& axes) {
    // Create appropriate gate structure matching qubit ordering.
    auto qubits = qubits_of(axes);
    assert(qubits.size() == 1);
    auto gate = qsim::Cirq::MatrixGate1<fp_type>::Create(0, qubits[0], matrix);
    // Apply gate
    auto state = active_state();
    active_simulator().ApplyGate(gate.qubits, gate.matrix.data(), state);
  }

  double norm_squared() {
    auto state = active_state();
    return active_state_space().Norm(state);
  }
 private:
  // Attributes
  State state_vec;
  const static std::vector<fp_type> swap_matrix;
  const unsigned num_threads;
  const unsigned max_qubits;
  std::unordered_map<std::string, std::list<unsigned>>
      axis_qubits; /**Qubits allocated to each axis.*/
  std::vector<std::string>
      qubit_axis; /** Axis assigned to each qubit. Inverse of axis_qubits.*/



  // Methods
  /**Number of qubits currently in memory.*/
  unsigned num_active_qubits() const { return qubit_axis.size(); }

};

template<typename Simulator>
const std::vector<typename KState<Simulator>::fp_type>
    KState<Simulator>::swap_matrix =
    qsim::Cirq::SWAP<KState::fp_type>::Create(0, 0, 1).matrix;

#endif //PROTOTYPE_STATE_REP_H_