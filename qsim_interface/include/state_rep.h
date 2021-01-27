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
#include "test_utils.h"

template<typename fp_type>
inline void match_to_reverse_qubits(std::vector<unsigned>& qubits,
                                    qsim::Matrix<fp_type>& matrix,
                                    std::vector<std::string>& qubit_axes);

template<typename T, typename Fun>
void static inline apply_permutation(
    std::vector<T>& v,
    std::vector<unsigned>& indices,
    Fun side_effect
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

  std::vector<std::string>
      qubit_axis; /** Axis assigned to each qubit. Inverse of axis_qubit.*/

  // Methods
  KState() = delete;

  KState(unsigned num_threads,
         unsigned max_qubits,
         const std::vector<std::string>& axis_labels
  )
      : state_vec(StateSpace(num_threads).Create(max_qubits)),
        num_threads(num_threads), max_qubits_(max_qubits),
        own_memory(true) {
    // Assign qubits to axis labels.
    for (auto axis_ptr = axis_labels.begin(); axis_ptr != axis_labels.end();
         ++axis_ptr) {
      add_qubit(*axis_ptr);
    }

    // Initialize state vector as zero state.
    StateSpace(num_threads).SetStateZero(state_vec);

  }

  /** Constructor using existing allocated memory for the state vector.
   *
   * Constructing KState in this way means it will not deallocated specified
   * memory if KState is deleted.
   * */
  template<typename StringIterable>
  KState(unsigned num_threads,
         unsigned max_qubits,
         const StringIterable axis_labels,
         fp_type* data
  )
      : state_vec(StateSpace(num_threads).Create(data, max_qubits)),
        num_threads(num_threads), max_qubits_(max_qubits),
        own_memory(false) {

    // Assign qubits to axis labels. Assume all axes are initially allocated one
    // qubit.
    for (auto axis_ptr = axis_labels.begin(); axis_ptr != axis_labels.end();
         ++axis_ptr) {
      add_qubit(*axis_ptr);
    }
  }

  /** Copy constructor*/
  KState(const KState& k_state) :
      qubit_axis(k_state.qubit_axis),
      state_vec(StateSpace(k_state.num_threads).Create(k_state.max_qubits_)),
      num_threads(k_state.num_threads),
      max_qubits_(k_state.max_qubits_),
      own_memory(true),
      axis_qubit(k_state.axis_qubit) {

    StateSpace(num_threads).Copy(k_state.state_vec, state_vec);
  }

  /** Move assignment */
  KState& operator=(KState&& other) noexcept {
    if (this != &other) {
      num_threads = other.num_threads;
      max_qubits_ = other.max_qubits_;
      state_vec = std::move(other.state_vec);
      axis_qubit = std::move(other.axis_qubit);
      qubit_axis = std::move(other.qubit_axis);
      own_memory = other.own_memory;
    }
    return *this;
  }

  /** Move constructor*/
  KState(KState&& k_state) noexcept
      : //We need to initialize state_vec because State has no trivial
  // constructor.
      state_vec(StateSpace::Null()) {
    *this = std::move(k_state);
  }

  /** \brief Copy another state into this one.
   *
   * @param source: KState to copy. Must have the same number of threads and
   *    max_qubits.
   * */
  void copy_from(KState& source) {
    if (max_qubits_ < source.max_qubits_) //Reallocation required.
      increment_max_qubits(source.max_qubits_ - max_qubits_);
    num_threads = source.num_threads;

    // Need to clear the state vector if it is larger than source's

    if (num_active_qubits() > source.num_active_qubits()) {
      State my_state = active_state();
      active_state_space().SetAllZeros(my_state);
    }

    //copy axis_qubit and qubit_axis
    axis_qubit = source.axis_qubit;
    qubit_axis = source.qubit_axis;
    assert(num_active_qubits() == source.num_active_qubits());

    //copy state vector
    State source_state = source.active_state();
    State my_state = active_state();
    StateSpace(num_threads).Copy(source_state, my_state);
  }

  /** Current maximum state vector size, in terms of stored qubits.*/
  size_t max_qubits() const { return max_qubits_; }

  /**Number of qubits currently in memory.*/
  inline unsigned num_active_qubits() const { return qubit_axis.size(); }

  /* State with current vector size.*/
  State active_state() {
    return StateSpace(num_threads).Create(state_vec.get(), num_active_qubits());
  }

  /** Apply swaps so that qubit axis assignment satisfy a desired order.
   *
   * This operation swaps axes in the system state so they are in the desired
   * order.
   *
   * After this method is called, the qubits are assigned such that reading axes
   * for each qubit (starting from qubit 0) produces a sequence agreeing with
   * the specified order.
   *
   * If more than one qubit is assigned to an axis, that qubit ordering is
   * preserved.
   *
   * */
  void order_axes(const std::vector<std::string>& ax_order) {

    std::unordered_map<std::string, unsigned> axis_index;
    for (size_t ii = 0; ii < ax_order.size(); ii++)
      axis_index.emplace(ax_order[ii], ii);

    // Determine new qubit order so that axes match desired ordering
    using AxisQubit = std::pair<std::string, unsigned>;
    std::vector<AxisQubit> sorted_axis_qubits;
    sorted_axis_qubits.reserve(qubit_axis.size());
    for (unsigned ii = 0; ii < qubit_axis.size(); ii++)
      sorted_axis_qubits.push_back({qubit_axis[ii], ii});

    std::sort(sorted_axis_qubits.begin(), sorted_axis_qubits.end(),
              [this, &axis_index](const AxisQubit& a, const AxisQubit& b) {
                return axis_index.at(a.first) < axis_index.at(b.first);
              });

    //Permutation required to sort qubit_axis
    std::vector<unsigned> permutation;
    permutation.reserve(qubit_axis.size());
    for (const auto& ax_q : sorted_axis_qubits)
      permutation.push_back(ax_q.second);

    //Apply qubit swaps and axis swaps so that new order is observed.
    auto effect = [this](unsigned q0, unsigned q1) { swap_qubits(q0, q1); };
    apply_permutation(qubit_axis, permutation, effect);

    //update axis_qubit
    axis_qubit.clear();
    for (unsigned q = 0; q < qubit_axis.size(); q++)
      axis_qubit[qubit_axis[q]] = q;
  }

  /** Allocate a qubit to an axis.
   *
   *
   * @param axis: The axis to which to add the qubit to. The axis should not
   *     previously have an allocated qubit.
   * */
  void add_qubit(const std::string& axis) {
    ASSERT(!axis_qubit.count(axis), "Tried to add a qubit to axis "
        << axis << " which is already allocated qubit " << axis_qubit.at(axis)
        << ".");
    assert(num_active_qubits() <= max_qubits_);
    if (num_active_qubits() == max_qubits_) //memory reallocation required.
      increment_max_qubits(1);

    axis_qubit.emplace(axis, num_active_qubits());
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
      q_and_ax.push_back({axis_qubit.at(ax), ax});
    }

    auto cmp = [this](const QubitAndAxis& a, const QubitAndAxis& b) {
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
  void transfer_qubit(const std::string& src, const std::string& dest) {

    assert(!axis_qubit.count(dest));
    ASSERT(axis_qubit.count(src), "Tried to move a qubit from "
        << src << ", but this axis is not allocated a qubit.");

    //reassign qubits from source to destination
    unsigned& qubit = axis_qubit[src];
    axis_qubit.emplace(dest, qubit);
    axis_qubit.erase(src);

    //Update qubit_axes
    qubit_axis[qubit] = dest;
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

  /** Whether an axis has been allocated a qubit.*/
  bool has_qubit(const std::string& axis) const {
    return axis_qubit.count(axis);
  }

  /** The qubit allocated to a given axis.*/
  unsigned qubit_of(const std::string& axis) const {
    ASSERT(axis_qubit.count(axis), "Tried to axis qubit for ("
        << axis << "), but none exists.\n");
    return axis_qubit.at(axis);
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

    //Handle trivial case - matrix acting on zero qubits. Global phase is not
    // observed but norm is adjusted.
    if (matrix.size() == 2) {
      const auto& x = matrix.at(0);
      const auto& y = matrix.at(1);
      rescale(sqrt(x * x + y * y));
    }

    // Qubits for each axis, in reverse order to account for qsim representation
    std::vector<unsigned> qubits;
    qubits.reserve(axes.size());
    for (auto q_ptr = axes.rbegin(); q_ptr != axes.rend(); ++q_ptr) {
      assert(axis_qubit.count(*q_ptr));
      qubits.push_back(axis_qubit.at(*q_ptr));
    }

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
  unsigned num_threads;
  unsigned max_qubits_;
  bool own_memory;
  using AxisQubit = std::unordered_map<std::string, unsigned>;
  AxisQubit axis_qubit; /**Qubits allocated to each axis.*/


  // Private methods
  /** Add another qubit to maximum memory size.
   *
   * This requires us to allocate more memory and copy the existing state vector
   * into it.
   * */
  void increment_max_qubits(size_t num_added) {
    assert (num_added > 0);
    assert (own_memory);

#ifdef DEBUG_SAMPLING
    std::cout << "Allocating more memory to include another qubit...\n";
    std::cout << "state vector before allocation:\n";
    print_amplitudes();
#endif

    max_qubits_ += num_added;
    StateSpace space = StateSpace(num_threads);

    // New memory
    State new_state_vec_max = space.Create(max_qubits_);
    space.SetAllZeros(new_state_vec_max);

    //Represent as active state
    State new_state_vec_current = space.Create(
        new_state_vec_max.get(), num_active_qubits());

    //Copy existing active state
    space.Copy(active_state(), new_state_vec_current);

    //Replace active_state
    state_vec = std::move(new_state_vec_max);

#ifdef DEBUG_SAMPLING
    std::cout << "state vector after allocation:\n";
    print_amplitudes();
#endif

  }

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
    ASSERT(axis_qubit.count(axis),
           "Tried to remove qubit of axis " << axis << ", but no qubit is"
                                                       " assigned to this axis.");

    // Swap the qubit of this axis with the last active qubit.
    const unsigned last_active_q = num_active_qubits() - 1;
    const unsigned removed_q = axis_qubit.at(axis);

    if (last_active_q != removed_q) {
      swap_qubits(removed_q, last_active_q);
      // Update the axis qubit registry for the axis involved in the swap.
      std::string axis_last = qubit_axis[last_active_q];
      assert(axis_qubit.count(axis_last));
      //q1 must be most recently added.
      assert(axis_qubit.at(axis_last) == last_active_q);
      axis_qubit.at(axis_last) = removed_q;
      qubit_axis[removed_q] = axis_last;
    }


    // Remove the axis.
    axis_qubit.erase(axis);
    qubit_axis.pop_back();

  }



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
template<typename T, typename Fun>
void static inline apply_permutation(
    std::vector<T>& v,
    std::vector<unsigned>& indices,
    Fun side_effect
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
