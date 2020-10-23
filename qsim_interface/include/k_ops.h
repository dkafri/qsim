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
        qubit_axes(qubit_axes),
        added_axes(added_axes),
        swap_sources(swap_sources),
        swap_sinks(swap_sinks),
        removed_axes(removed_axes) {
    //Validation
    assert(powl(2, 1 + 2 * qubit_axes.size()) == matrix.size());
    // Elements of added_axes must be in qubit_axes an appropriate number of 
    // times.
    size_t count;
    auto qaxes_end = qubit_axes.end();
    auto sources_end = swap_sources.end();
    for (const auto& axis: added_axes) {
      count = std::count(added_axes.begin(), added_axes.end(), axis);
      assert(count == std::count(qubit_axes.begin(), qaxes_end, axis));
    }
    // Elements of swap_sources must be unique and contained in qubit_axes and
    // not overlap with removed_axes.
    for (const auto& axis: swap_sources) {
      count = std::count(swap_sources.begin(), swap_sources.end(), axis);
      assert(count == 1);
      assert(std::find(qubit_axes.begin(), qaxes_end, axis) != qaxes_end);
      assert(std::find(swap_sources.begin(), sources_end, axis) == sources_end);
    }

    // Elements of removed_axes must be unique and contained in qubit_axes
    for (const auto& axis: removed_axes) {
      count = std::count(removed_axes.begin(), removed_axes.end(), axis);
      assert(count == 1);
      assert(std::find(qubit_axes.begin(), qaxes_end, axis) != qaxes_end);
    }

    assert(swap_sinks.size() == swap_sources.size());

  };

};

//A single channel, equivalent to Cirq definition of a Channel
template<typename fp_type>
using KChannel = std::vector<KOperator<fp_type>>;

///** Representation of a KrausOperation*/
//template<typename fp_type, size_t num_qubits>
//struct KOperation {
//  using KChannel = KChannel<fp_type, num_qubits>;
//
//  std::unordered_map<const std::vector<size_t>, KChannel>
//      channels; /** Mapping between conditional register values and corresponding channels.*/
//  std::vector<std::string> conditional_registers; /**
// * Classical registers on which the operation is conditioned. Each key of
// * channels should have the same length as conditional_registers. The
// * register values are referenced in the specified order.*/
//  bool is_measurement = false; /** Whether operation is a measurement.
// * Sampled measurements are recorded as classical registers.*/
//  std::string label = "unlabeled operation"; /** Operation label. If this
// * operation is a measurement, a register with this label is created when
// * this operation is sampled.*/
//  bool is_virtual = false; /** Whether the operation is virtual. Virtual
// * operations have a temporary effect on the simulation and are back-tracked
// * when a non-virtual operation is met.*/
//
//  /** Return the channel conditioned on current values of classical registers.*/
//  KChannel channel_at(std::unordered_map<std::string, size_t>& registers) {
//    std::vector<size_t> reg_vals;
//    for (const auto& reg : conditional_registers) {
//      reg_vals.push_back(registers.at(reg));
//    }
//    return channels.at(reg_vals);
//  }
//
//  /** Construct a KOperation without conditional registers. */
//  static KOperation unconditioned(const std::vector<KChannel>& channel,
//                                  bool is_measurement = false,
//                                  bool is_virtual = false) {
//    std::unordered_map<const std::vector<size_t>, KChannel> channels;
//    channels.emplace({}, channel);
//    return KOperation<fp_type, num_qubits>{channels, {}, is_measurement,
//                                           is_virtual};
//  }
//};

#endif //PROTOTYPE_INCLUDE_Q_OPS_H_