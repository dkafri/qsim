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
#include <functional>
#include "../../lib/matrix.h"
#include "../../lib/gates_cirq.h"

template<typename fp_type>
inline void match_to_reverse_qubits(std::vector<unsigned>& qubits,
                                    qsim::Matrix<fp_type>& matrix,
                                    std::vector<std::string>& qubit_axes);

template<typename T>
void static inline apply_permutation(
    std::vector<T>& v,
    std::vector<unsigned>& indices,
    const std::function<void(unsigned, unsigned)>& side_effect
);

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

  /* State with current vector size.*/
  State active_state() {
    return StateSpace(num_threads).Create(state_vec.get(), num_active_qubits());
  }

  /** Sort qubit axes in alphabetical axis order, reverse qubits.
   *
   * This operation swaps axes in the system state so they are in alphabetical
   * order. It also reverses the qubit ordering so that it matches the typical
   * numpy.kron definition (i.e. the last qubit index changes fastest).
   *
   * After this method is called, the qubits are assigned such that reading
   * qubits of all axes in alphabetical order produces 0,1,2,3,...
   *
   * Once this method is called the qubit order no longer matches what qsim
   * expects, so applying matrices to the state will not work as expected.
   *
   * */
  void c_align() {
    //TODO: Proper testing of this method when axes are assigned more than 1
    // qubit.

    // Determine new qubit order so that axes are alphabetical
    using AxisQubit = std::pair<std::string, unsigned>;
    std::vector<AxisQubit> sorted_axes_qubits;
    sorted_axes_qubits.reserve(qubit_axis.size());
    for (unsigned ii = 0; ii < qubit_axis.size(); ii++)
      sorted_axes_qubits.push_back({qubit_axis[ii], ii});

    std::sort(sorted_axes_qubits.begin(), sorted_axes_qubits.end(),
              [](const AxisQubit& a, const AxisQubit& b) {
                if (a.first != b.first)
                  return a.first > b.first;
                return a.second > b.second;
              });

    //Permutation required to sort qubit_axis alphabetically
    std::vector<unsigned> permutation;
    permutation.reserve(qubit_axis.size());
    for (const auto& ax_q : sorted_axes_qubits)
      permutation.push_back(ax_q.second);

    //Apply qubit swaps and axis swaps so that new order is observed.
    std::function<void(unsigned, unsigned)>
        effect = [this](unsigned q0, unsigned q1) { swap_qubits(q0, q1); };
    apply_permutation(qubit_axis, permutation, effect);
    //reverse the qubit assignment to match C ordering
    std::reverse(qubit_axis.begin(), qubit_axis.end());


    //update axis_qubits
    axis_qubits.clear();
    for (unsigned q = 0; q < qubit_axis.size(); q++)
      axis_qubits[qubit_axis[q]].push_back(q);

  }

  /** Reverse qubit order to match qsim.
   *
   * This method partially undoes the action of c_align() by reassigning
   * qubits to axes in reverse order. After this is invoked matrix operations
   * may be applied on the system state again.
   *
   * This operation is efficient in that it only reassigned qubits to axes in
   * reverse order and does not require any manipulation of the state vector.
   *
   * */
  void f_align() {

    //reverse qubit_axis
    std::reverse(qubit_axis.begin(), qubit_axis.end());
    //update axis_qubits
    axis_qubits.clear();
    for (unsigned q = 0; q < qubit_axis.size(); q++)
      axis_qubits[qubit_axis[q]].push_back(q);

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
          //a qubit was added first, so it should be removed last
          if (q == a.first)
            return false;
          //b qubit was added first, so a's qubit should be removed first
          if (q == b.first)
            return true;
        }
      }
      // Axis whose qubit is higher in the order should be removed first.
      return a.first > b.first;
    };
    q_and_ax.sort(cmp);

    for (const auto& pair: q_and_ax) {
      remove_qubit(pair.second);
    }

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

#ifdef DEBUG_SAMPLING
  void print_qubit_axes() {
    for (const auto& ax: qubit_axis)
      std::cout << "(" << ax  << "),";
    std::cout << std::endl;
  }
#endif

  /** Rescale the state vector by a constant.*/
  void rescale(double scale) {
    auto state = active_state();
    active_state_space().Multiply(scale, state);
  }

  /** All qubits allocated to a given axis.*/
  std::vector<unsigned> qubits_of(const std::string& axis) {
    const auto& qubits = axis_qubits[axis];
    return std::vector<unsigned>(qubits.begin(), qubits.end());
  }

  /** Permute and apply a matrix to the specified axes.
   *
   * Before applying a matrix, this operation applies a permutation to it
   * according to the qubits assigned to the given axes. It also permutes the
   * axes accordingly.
   *
   * @param matrix: Matrix to apply, stored as alternating real and imaginary
   * parts.
   * @param axes: Order of axes corresponding to the qubits of the matrix. Must
   * satisfy matrix.size() == 2^(2*axes.size())
 * */
  void permute_and_apply(qsim::Matrix<fp_type>& matrix,
                         std::vector<std::string>& axes) {

    // Qubits for each axis, in reverse order to account for qsim representation
    auto qubits = qubits_vec(axes, true);

    match_to_reverse_qubits(qubits, matrix, axes);

    // Now we can finally apply the matrix
    auto state = active_state();
    active_simulator().ApplyGate(qubits, matrix.data(), state);
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
  using AxisQubits = std::unordered_map<std::string, std::list<unsigned>>;
  AxisQubits axis_qubits; /**Qubits allocated to each axis.*/
  std::vector<std::string>
      qubit_axis; /** Axis assigned to each qubit. Inverse of axis_qubits.*/

  // Private methods
  /** Simulator appropriate to current vector size.*/
  inline Simulator active_simulator() const { return Simulator(num_threads); }

  /** State space appropriate to current vector size.*/
  inline StateSpace active_state_space() const {
    return StateSpace(num_threads);
  }

  /** Swap two qubits in the state vector.
   *
   * @param q0 : Any active qubit index.
   * @param q1: Another active qubit index.
   * */
  inline void swap_qubits(unsigned int q0, unsigned int q1) {

    static std::vector<unsigned> qubits(2);
    if (q0 > q1)
      qubits = {q1, q0};
    else
      qubits = {q0, q1};

    auto state = active_state();
    active_simulator().ApplyGate(qubits, swap_matrix.data(), state);

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
      swap_qubits(removed_q, last_active_q);
      // Update the axis qubit registry for the axis involved in the swap.
      std::string axis_1 = qubit_axis[last_active_q];
      //q1 must be most recently added.
      assert(axis_qubits[axis_1].back() == last_active_q);
      axis_qubits[axis_1].pop_back();
      axis_qubits[axis_1].push_back(removed_q);
      qubit_axis[removed_q] = axis_1;
    }


    // Deallocate the removed qubit.
    axis_qubits[axis].pop_back();
    qubit_axis.pop_back();

  }
  /** Qubits allocated to one or more axes, accounting for repeats
   *
   * Returns a vector of qubit indices with exactly the same length as the
   * input. If an axis is allocated more than one qubit, it should be included
   * in the input that many times.
   *
   * @param axes - Sequence of axis labels. May contain repeats.
   * @param reverse - Whether the qubits should be returned in reverse order
   *     (this reflects qsim's ordering).
   *
   * Example:
   * Axis "a" is allocated qubits [2,0]
   * and axis "b" is allocated qubits [1,3]
   *
   * Input ({"a","b","a"},reverse=false) returns [2,1,0]
   * Input ({"a","b","a},reverse=true) returns [0,3,2]
   * */
  std::vector<unsigned> qubits_vec(const std::vector<std::string>& axes,
                                   bool reverse = false) {
    std::vector<unsigned> out{};
    out.reserve(axes.size());

    if (reverse) {
      using IterType=AxisQubits::mapped_type::const_reverse_iterator;

      std::unordered_map<std::string, IterType> iters;
      IterType iter;

      for (auto axis_ptr = axes.rbegin(); axis_ptr != axes.rend(); ++axis_ptr) {
        //get current qubit iterator for axis
        if (iters.count(*axis_ptr)) iter = iters.at(*axis_ptr);
        else iter = axis_qubits[*axis_ptr].rbegin();

        assert(iter != axis_qubits[*axis_ptr].rend());
        out.push_back(*iter);
        iters[*axis_ptr] = ++iter;
      }

    } else {
      using IterType=AxisQubits::mapped_type::const_iterator;

      std::unordered_map<std::string, IterType> iters;
      IterType iter;

      for (const auto& axis : axes) {
        //get current qubit iterator for axis
        if (iters.count(axis)) iter = iters.at(axis);
        else iter = axis_qubits[axis].cbegin();

        assert(iter != axis_qubits[axis].cend());
        out.push_back(*iter);
        iters[axis] = ++iter;
      }

    }

    return out;
  }

  /**Number of qubits currently in memory.*/
  inline unsigned num_active_qubits() const { return qubit_axis.size(); }

};

template<typename Simulator>
const std::vector<typename KState<Simulator>::fp_type>
    KState<Simulator>::swap_matrix =
    qsim::Cirq::SWAP<KState::fp_type>::Create(0, 0, 1).matrix;

/** Apply required axis and matrix permutations to bring the matrix to
 * qsim normal order.
 *
 * This function does nothing if the axes are already in reverse qubit order.
 * If they are not, then we apply swap gates to the matrix and permute the
 * corresponding qubit_axes (and the qubits vector so that it is ascending).
 *
 * @param qubits: The qubits assigned to each of qubit_axes, in reverse
 *     order.
 * */
template<typename fp_type>
inline void match_to_reverse_qubits(std::vector<unsigned>& qubits,
                                    qsim::Matrix<fp_type>& matrix,
                                    std::vector<std::string>& qubit_axes) {

  // qubits must be in increasing order in order to apply the matrix
  // correctly. To account for this we need to permute the qubits and
  // accordingly permute the matrix.
  std::vector<unsigned> perm = qsim::NormalToGateOrderPermutation(qubits);
  if (!perm.empty()) { //Only permute if permutation is non-trivial.
    // Apply swaps to the matrix that reorders qubits
    qsim::MatrixShuffle(perm, qubits.size(), matrix);
    std::sort(qubits.begin(), qubits.end());

    //Also permute axes. These are in reverse order.
    std::vector<std::string> new_axes;
    auto size = qubit_axes.size();

    new_axes.reserve(size);
    // perm was defined for qubits in reverse order, so we reverse axes first,
    // apply the permutation, then reverse back.
    std::reverse(qubit_axes.begin(), qubit_axes.end());
    for (const auto& ind :perm) new_axes.push_back(qubit_axes[ind]);
    std::reverse(new_axes.begin(), new_axes.end());
    qubit_axes = std::move(new_axes);
  }
}

// Apply a permutation to a vector using swap operations. For each swap between
// two indices in the vector, apply a side effect conditioned on those indices.
// Source: https://devblogs.microsoft.com/oldnewthing/20170102-00/?p=95095
template<typename T>
void static inline apply_permutation(
    std::vector<T>& v,
    std::vector<unsigned>& indices,
    const std::function<void(unsigned, unsigned)>& side_effect
) {
  using std::swap; // to permit Koenig lookup
  for (unsigned i = 0; i < indices.size(); i++) {
    unsigned current = i;
    while (i != indices[current]) {
      unsigned next = indices[current];
      side_effect(current, next);
      swap(v[current], v[next]);
      indices[current] = current;
      current = next;
    }
    indices[current] = current;
  }
}

#endif //PROTOTYPE_STATE_REP_H_
