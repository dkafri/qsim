//
// Created by dkafri on 10/12/20.
//
#include <iostream>
#include <random>
#include <k_ops.h>
#include "include/state_rep.h"
#include "include/sampling.h"

#include "../lib/formux.h"
#include "../lib/simmux.h"

using namespace std;


int main() {

  using Simulator = qsim::Simulator<qsim::For>;
  using fp_type = Simulator::fp_type;
  using complex_type = complex<fp_type>;

  random_device rd;
  std::mt19937 rgen(rd());
  unordered_map<string, size_t> registers;

  vector<string> axis_labels = {"a", "b"};
  KState<Simulator> k_state(3, 3, axis_labels);
  KState<Simulator> tmp_state(k_state);

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

  auto prep_c = KOperation<fp_type>::unconditioned({q_op});
  double cutoff = qsim::RandomValue(rgen, 1.0);
  sample_op(prep_c, k_state, tmp_state, registers, cutoff);

  cout << "after setting c in plus state:\n";
  k_state.print_amplitudes();

  //Apply CNOT between c and b
  KOperator<fp_type> CNOT_op{{1, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 1, 0,
                              0, 0, 0, 0, 1, 0, 0, 0},
                             {}, {"c", "b"}, {}, {},
                             {}};
  auto CNOT = KOperation<fp_type>::unconditioned({CNOT_op});

  cutoff = qsim::RandomValue(rgen, 1.0);
  sample_op(CNOT, k_state, tmp_state, registers, cutoff);

  cout << "After CNOT from c to b\n";
  k_state.print_amplitudes();

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



  // Do sampling
  sample_op(op, k_state, tmp_state, registers, cutoff);

  cout << "after measuring b in the X basis\n";
  cout << "registers:\n";
  for (const auto& key_val: registers)
    cout << key_val.first << ": " << key_val.second << endl;
  k_state.print_amplitudes();

  return 0;
}
