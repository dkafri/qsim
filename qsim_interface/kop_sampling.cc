//
// Created by dkafri on 10/12/20.
//
#include <iostream>
#include <random>
#include <k_ops.h>
#include "include/state_rep.h"
#include "include/k_ops.h"
#include "include/sampling.h"

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

  auto op = KOperation<fp_type>::unconditioned({q_op0, q_op1},
                                               true,
                                               "measure b",
                                               false);
  unordered_map<string, size_t> registers;

  random_device rd;
  std::mt19937 rgen(rd());
  double cutoff = qsim::RandomValue(rgen, 1.0);

  // Do sampling
  sample_op(op, k_state, tmp_state, registers, cutoff);

  cout << "final registers:\n";
  for (const auto& key_val: registers)
    cout << key_val.first << ": " << key_val.second << endl;

  cout << "final amplitudes:\n";
  k_state.print_amplitudes();

  return 0;
}
