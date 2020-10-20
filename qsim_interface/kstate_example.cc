//
// Created by dkafri on 10/12/20.
//
#include <iostream>
#include <random>
#include "include/state_rep.h"
#include "../lib/formux.h"
#include "../lib/simmux.h"

using namespace std;
int main() {

  using Simulator = qsim::Simulator<qsim::For>;
  using StateSpace = Simulator::StateSpace;
  using fp_type = Simulator::fp_type;

  unsigned num_threads = 3;

  vector<string> axis_labels = {"a", "b"};
  KState<Simulator> k_state(num_threads, 3, axis_labels);

  for (const auto& axis :axis_labels) {
    cout << "qubits assigned to " << axis << ": ";
    for (const auto& q:k_state.qubits_of(axis))
      cout << q << ", ";
    cout << endl;
  }
  cout << "initial 2 qubit state:\n";
  k_state.print_amplitudes();

  cout << "adding a qubit to c:\n";
  // Try to create axis "c" in state +
  // First allocate a qubit
  k_state.add_qubit("c");

  // Define the 2x2 effective matrix with first column corresponding to desired
  // state. Real and imaginary parts of the matrix are stored separately, in
  // alternating order (R[0,0],I[0,0],R[0,1],I[0,1],...)
  constexpr fp_type sqrt_half = 0.7071067811865476;
  constexpr fp_type k_op[] = {sqrt_half, 0, 0, 0, sqrt_half, 0, 0, 0};

  auto qs = k_state.qubits_of("c");
  cout << "qubits assigned to c: ";
  for (const auto& q:qs)
    cout << q << ", ";
  cout << endl;

  auto state = k_state.active_state();

  Simulator(num_threads).ApplyGate(qs, k_op, state);

  cout << "state after adding c and setting it to +:\n";
  k_state.print_amplitudes();

  auto qubits = k_state.qubits_of(vector<string>{"c", "a"});

  //Apply a CNOT from c to a
  auto CNOT = qsim::Cirq::CNOT<fp_type>::Create(0, qubits[0], qubits[1]).matrix;

  cout << "CNOT from c to a:\n";
  state = k_state.active_state();
  cout << "target qubits:\n";
  for (const auto& q: qubits)
    cout << q << ", ";
  cout << endl;
  Simulator(num_threads).ApplyGate(qubits, CNOT.data(), state);
  k_state.print_amplitudes();


  // Measure "c" in computational basis. This corresponds to two 1x2 matrices,
  // which we pad with a row of zeros. This is not the most efficient approach.
  constexpr fp_type k_ops[2][8]{{1, 0, 0, 0, 0, 0, 0, 0},
                                {0, 0, 1, 0, 0, 0, 0, 0}};

  KState<Simulator> tmp_state(k_state);

  random_device rd;
  std::mt19937 rgen(rd());

  double cutoff = qsim::RandomValue(rgen, 1.0);
  double norm2 = 0.0;

  unsigned ii;
  for (ii = 0; ii < 2; ii++) {
    state = k_state.active_state();
    Simulator(num_threads).ApplyGate(k_state.qubits_of("c"),
                                     k_ops[ii], state);
    norm2 = StateSpace(num_threads).Norm(state);
    cutoff -= norm2;
    if (cutoff < 0)
      break;
    k_state.copy_from(tmp_state);
  }
  k_state.remove_qubit("c");
  state = k_state.active_state();
  StateSpace(num_threads).Multiply(1 / sqrt(norm2), state);

  cout << "destructive measurement of c:\n";
  cout << "sampled index: " << ii << endl;
  cout << "resulting state:\n";
  k_state.print_amplitudes();

  return 0;
}
