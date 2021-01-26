//
// Created by dkafri on 10/15/20.
//

#ifndef PROTOTYPE_INCLUDE_Q_OPS_H_
#define PROTOTYPE_INCLUDE_Q_OPS_H_

#include <utility>
#include <vector>
#include <string>
#include <gates_cirq.h>
#include <cassert>
#include <unordered_map>
#include <map>

/** Representation of a Kraus operator as a square qubit matrix.
 *
 * Notes:
 * Non-square Kraus operators can be represented by padding either their rows
 * or columns with zeros. (their original shapes must be powers of 2.) The row
 * and column indices can be viewed as multi-indices running over multiple
 * qubits. I.e. the row index of a 4x4 matrix is actual a multi-index running
 * over 2 qubits. Importantly, the qubit order of the row multi-index is assumed
 * to be the same as that for the column multi-index.
 *
 * A non-square Kraus operator with more rows than columns corresponds to an
 * operation that increases the Hilbert space dimension of at least one axis.
 * This is done by adding qubits to that axis. The square matrix representation
 * of this operation is obtained by padding 0 columns to the Kraus operator.
 * Importantly, the row multi-index should be ordered so that the added qubits
 * are at the end. Similarly, for Kraus operators that remove qubits (lower
 * axis dimension), the multi-index should be ordered so that the removed axes
 * are at the end.
 *
 * If we have to apply a matrix whose axis qubit order does not match the
 * existing state qubit order, we need to first permute it. But then we can
 * just permute it back in order to maintain correct axis order. For the moment
 * this only applies to 4x4 matrices since qsim only handles at most 2-qubit
 * gates.
 * */
template<typename fp_type>
struct KOperator {
  using MatrixType = qsim::Matrix<fp_type>;

  MatrixType matrix; /** Square matrix representation of a Kraus operator.*/
  std::vector<std::string> added_axes; /** Which elements of qubit_axes need 
 * qubit allocations before applying the matrix. All elements must be elements
 * of qubit_axes and should appear the same number of times. A qubit is added
 * to an axis once for each time it appears.*/
  std::vector<std::string> qubit_axes; /**
 * Axes associated with each qubit of the row/matrix multi-index. Axes should be
 * included as many times as qubits they require.*/
  std::vector<std::string> swap_sources; /**
 * Axes whose qubits should be transferred to another axis after the matrix is
 * applied. All elements must be unique and contained in qubit_axes.*/
  std::vector<std::string> swap_sinks; /**
 * Axes receiving qubits from the corresponding swap sources. All elements must
 * be unique. These axes should not have allocated qubits before the swap 
 * operation. Must not overlap with qubit_axis.*/
  std::vector<std::string> removed_axes; /** Which axes have their qubits 
 * removed after the KOperator matrix is applied and swaps occur. Must not
 * overlap with swap_sources. All elements must be unique and contained in
 * qubit_axes. Any removed axis has all of its qubits deallocated.*/

  KOperator(const MatrixType& matrix,
            const std::vector<std::string>& added_axes,
            const std::vector<std::string>& qubit_axes,
            const std::vector<std::string>& swap_sources,
            const std::vector<std::string>& swap_sinks,
            const std::vector<std::string>& removed_axes
  )
      : matrix(matrix),
        added_axes(added_axes),
        qubit_axes(qubit_axes),
        swap_sources(swap_sources),
        swap_sinks(swap_sinks),
        removed_axes(removed_axes) {
    //Validation
    size_t lhs = size_t{1} << (1 + 2 * qubit_axes.size());
    if (lhs != matrix.size())
      throw std::runtime_error(
          "Matrix size " + std::to_string(matrix.size())
              + " does not match 2 ^ (1 + 2 * qubit_axes.size()) = "
              + std::to_string(lhs));

    // Elements of added_axes must be in qubit_axes an appropriate number of 
    // times.
    size_t count;
    auto qaxes_end = qubit_axes.end();
    auto sources_end = swap_sources.end();
    for (const auto& axis: added_axes) {
      count = std::count(added_axes.begin(), added_axes.end(), axis);
      size_t rhs = std::count(qubit_axes.begin(), qaxes_end, axis);
      if (count != rhs)
        throw std::runtime_error(" axis " + axis + " appears "
                                     + std::to_string(count)
                                     + " times in added_axes and "
                                     + std::to_string(rhs)
                                     + " times in qubit_axes.");
    }
    // Elements of swap_sources must be unique and contained in qubit_axes and
    // not overlap with removed_axes.
    for (const auto& axis: swap_sources) {
      count = std::count(swap_sources.begin(), sources_end, axis);
      assert(count == 1);
      assert(std::find(qubit_axes.begin(), qaxes_end, axis) != qaxes_end);
      assert(std::find(removed_axes.begin(), removed_axes.end(), axis)
                 == removed_axes.end());
    }

    // Elements of removed_axes must be unique and contained in qubit_axes
    for (const auto& axis: removed_axes) {
      count = std::count(removed_axes.begin(), removed_axes.end(), axis);
      assert(count == 1);
      assert(std::find(qubit_axes.begin(), qaxes_end, axis) != qaxes_end);
    }

    assert(swap_sinks.size() == swap_sources.size());

  };

#ifdef DEBUG_SAMPLING
  /** Print to cout for debugging purposes*/
  void print() const {

    std::cout << " Matrix: (";
    for (size_t ii = 0; ii < matrix.size() / 2; ii++)
      std::cout << matrix[2 * ii] << "+i*" << matrix[2 * ii + 1] << ",";
    std::cout << ")";
    if (added_axes.size()) {
      std::cout << ", added: (";
      for (const auto& ax : added_axes)
        std::cout << ax << ", ";
      std::cout << "), ";
    }
    if (qubit_axes.size()) {
      std::cout << "qubit_axes: (";
      for (const auto& ax : qubit_axes)
        std::cout << ax << ", ";
      std::cout << "), ";
    }
    if (swap_sources.size()) {
      std::cout << "swap_sources: (";
      for (const auto& ax : swap_sources)
        std::cout << ax << ", ";
      std::cout << "), ";
    }
    if (swap_sinks.size()) {
      std::cout << "swap_sinks: (";
      for (const auto& ax : swap_sinks)
        std::cout << ax << ", ";
      std::cout << "), ";
    }
    if (removed_axes.size()) {
      std::cout << "removed_axes: (";
      for (const auto& ax : removed_axes)
        std::cout << ax << ", ";
      std::cout << ")";

    }
  }
#endif

};

//A single channel, equivalent to Cirq definition of a Channel
template<typename fp_type>
using KChannel = std::vector<KOperator<fp_type>>;

///** Representation of a KrausOperation*/
template<typename fp_type>
struct KOperation {
  using ChannelMap=std::map<std::vector<size_t>, KChannel<fp_type>>;
  ChannelMap channels; /** Mapping between conditional register values and 
 * corresponding channels.*/
  std::vector<std::string> conditional_registers; /**
 * Classical registers on which the operation is conditioned. Each key of
 * channels should have the same length as conditional_registers. The
 * register values are referenced in the specified order.*/
  bool is_recorded; /** Whether sampling result should be recorded.*/
  std::string label; /** Operation label. If this
 * operation is recorded, a register with this label is created when
 * this operation is sampled.*/
  bool is_virtual; /** Whether the operation is virtual. Virtual
 * operations have a temporary effect on the simulation and are back-tracked
 * when a non-virtual operation is met.*/


  /** Return the channel conditioned on current values of classical registers.*/
  KChannel<fp_type> channel_at(std::unordered_map<std::string,
                                                  size_t>& registers) {

    std::vector<size_t> reg_vals;
    for (const auto& reg : conditional_registers) {
      assert(registers.count(reg));
      reg_vals.push_back(registers.at(reg));

    }

    assert(channels.count(reg_vals));
    return channels.at(reg_vals);
  }

  /** Single operator constructor */
  explicit KOperation(const KOperator<fp_type>& k_op,
                      bool is_recorded = false,
                      std::string label = "unlabeled op",
                      bool is_virtual = false)
      : channels{{{}, {k_op}}},
        conditional_registers{},
        is_recorded(is_recorded),
        label(std::move(label)),
        is_virtual(is_virtual) {}

  /** Unconditioned operation constructor */
  explicit KOperation(const KChannel<fp_type>& channel,
                      bool is_recorded = false,
                      std::string label = "unlabeled op",
                      bool is_virtual = false)
      : channels{{{}, channel}},
        conditional_registers{},
        is_recorded(is_recorded),
        label(std::move(label)),
        is_virtual(is_virtual) {}

  /** General constructor with conditional registers*/
  explicit KOperation(const ChannelMap& channel_map,
                      std::vector<std::string> conditional_registers,
                      bool is_recorded = false,
                      std::string label = "unlabeled op",
                      bool is_virtual = false)
      : channels(channel_map),
        conditional_registers(std::move(conditional_registers)),
        is_recorded(is_recorded),
        label(std::move(label)),
        is_virtual(is_virtual) {}

#ifdef DEBUG_SAMPLING
  /** print method for debugging purposes*/
  void print() const {
    std::cout << "KOperation (" << label << "):\n";
    std::cout << "conditional registers: ";
    for (auto& creg : conditional_registers)
      std::cout << creg << ", ";
    std::cout << std::endl;

    std::cout << "channels:\n";
    for (const auto& k_v : channels) {
      std::cout << "\t(";
      for (const auto& ii : k_v.first)
        std::cout << ii << ",";
      std::cout << "): ";
      for (const auto& k_op: k_v.second)
        k_op.print();
      std::cout << std::endl;

    }
    std::cout << "\nis_recorded: " << is_recorded << ", ";
    std::cout << "is_virtual: " << is_virtual << std::endl;
  }

#endif
};

#endif //PROTOTYPE_INCLUDE_Q_OPS_H_
