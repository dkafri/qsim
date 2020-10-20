//
// Created by dkafri on 10/12/20.
//
#include <iostream>
#include <random>
#include <q_ops.h>
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
  constexpr complex_type sqrt_half = 0.7071067811865476;

  qsim::Cirq::Matrix1q<fp_type> matrix1{sqrt_half, 0, sqrt_half, 0};

  QOperator<fp_type, 1> q_op{matrix1, {"c"}, {"c"}, {}, {}, {}};

  k_state.add_qubits_for(q_op);
  k_state.apply(q_op.matrix, q_op.qubit_axes);

  cout << "after setting c in pluse state:\n";
  k_state.print_amplitudes();

  KState<Simulator> tmp_state(k_state);
  // Measure b in + basis stochastically.
  QOperator<fp_type, 1>
      q_op0{{sqrt_half, sqrt_half, 0, 0}, {}, {"b"}, {}, {}, {"b"}};
  QOperator<fp_type, 1>
      q_op1{{sqrt_half, -sqrt_half, 0, 0}, {}, {"b"}, {}, {}, {"b"}};

  unsigned k_ind = 0;
  double norm2 = 1.0;
  random_device rd;
  std::mt19937 rgen(rd());
  double cutoff = qsim::RandomValue(rgen, 1.0);

  for (auto& k_op: {q_op0, q_op1}) {
    k_state.add_qubits_for(k_op);
    tmp_state.add_qubits_for(k_op);

    k_state.apply(k_op.matrix, k_op.qubit_axes);

    norm2 = k_state.norm_squared();
    cutoff -= norm2;
    if (cutoff < 0)
      // Apply swaps
      for (auto ii = 0; ii < k_op.swap_sources.size(); ii++) {
        k_state.transfer_qubits(k_op.swap_sources[ii], k_op.swap_sinks[ii]);
      }
    // Remove axes
    k_state.remove_qubits_of(k_op.removed_axes);

    break;
    k_state.copy_from(tmp_state);
  }

  return 0;
}