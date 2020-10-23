//
// Created by dkafri on 10/12/20.
//
#include <iostream>
#include <random>
#include <k_ops.h>
#include "include/state_rep.h"
#include "../lib/formux.h"
#include "../lib/simmux.h"

using namespace std;

size_t log2_b(size_t number) {
  size_t out = 0;

  while (number >>= size_t{1}) ++out;

  return out;
}

int main() {

  using Simulator = qsim::Simulator<qsim::For>;
  using fp_type = Simulator::fp_type;
  using complex_type = complex<fp_type>;

  vector<string> axis_labels = {"a", "b"};
  KState<Simulator> k_state(3, 3, axis_labels);

  for (const auto& axis :axis_labels) {
    cout << "qubits assigned to " << axis << ": ";
    for (const auto& q:k_state.qubits_of(axis))
      cout << q << ", ";
    cout << endl;
  }
  cout << "initial 2 qubit state:\n";
  k_state.print_amplitudes();

  // Prepare axis "c" in the plus state using a QOperation.
  constexpr fp_type sqrt_half = 0.7071067811865476;

  KOperator<fp_type> q_op{{sqrt_half, 0, 0, 0, sqrt_half, 0, 0, 0},
                          {"c"}, {"c"}, {}, {}, {}};

  for (const auto& ax: q_op.added_axes) k_state.add_qubit(ax);
  k_state.permute_and_apply(q_op.matrix, q_op.qubit_axes);

  cout << "after setting c in plus state:\n";
  k_state.print_amplitudes();

  //Apply CNOT between c and b
  qsim::Matrix<fp_type> CNOT_mat{1, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 1, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 1, 0,
                                 0, 0, 0, 0, 1, 0, 0, 0};
  vector<string> axes{"c", "b"};
  k_state.permute_and_apply(CNOT_mat, axes);

  cout << "After CNOT from c to b\n";
  k_state.print_amplitudes();

  KState<Simulator> tmp_state(k_state);
  // Measure b in + basis stochastically.
  KOperator<fp_type>
      q_op0{{sqrt_half, 0, sqrt_half, 0, 0, 0, 0, 0}, {}, {"b"}, {}, {}, {"b"}};
  KOperator<fp_type>
      q_op1
      {{sqrt_half, 0, -sqrt_half, 0, 0, 0, 0, 0}, {}, {"b"}, {}, {}, {"b"}};
  KChannel<fp_type> channel{q_op0, q_op1};

  unsigned k_ind = 0;
  double norm2;
  random_device rd;
  std::mt19937 rgen(rd());
  double cutoff = qsim::RandomValue(rgen, 1.0);

  for (auto& k_op : channel) {
    //Add required axes
    for (const auto& ax: k_op.added_axes) {
      k_state.add_qubit(ax);
      tmp_state.add_qubit(ax);
    }

    //Permute and apply matrix
    k_state.permute_and_apply(k_op.matrix, k_op.qubit_axes);

    norm2 = k_state.norm_squared();
    cout << "sampling norm: " << norm2 << endl;
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

  cout << "sampled index: " << k_ind << endl;
  cout << "final amplitudes:\n";
  k_state.print_amplitudes();

  return 0;
}
