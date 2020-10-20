//
// Created by dkafri on 10/20/20.
//


#include <gates_cirq.h>
#include <simmux.h>
#include <formux.h>
#include <gate_appl.h>
#include <iostream>

using namespace std;
using Simulator = qsim::Simulator<qsim::For>;
using StateSpace = Simulator::StateSpace;
using State = StateSpace::State;
using fp_type = Simulator::fp_type;

void print_amplitudes(State& state) {

  for (size_t ii = 0; ii < powl(2, state.num_qubits()); ii++)
    cout << StateSpace::GetAmpl(state, ii) << ", ";
  cout << endl;

}

int main() {

  unsigned num_threads = 4;
  unsigned num_qubits = 2;
  Simulator simulator(num_threads);
  StateSpace space(num_threads);
  State state = StateSpace::Create(2);

  //Set to |00>
  StateSpace::SetAmpl(state, 0, 1, 0);
  print_amplitudes(state);

  //Apply X on 0, expect |10>
  auto X = qsim::Cirq::X<fp_type>::Create(0, 0);
  qsim::ApplyGate(simulator, X, state);
  print_amplitudes(state);

  //Apply CNOT 0->1, expect |11>
  qsim::Cirq::Matrix2q<Simulator::fp_type> CNOT_mat{1, 0, 0, 0,
                                                    0, 0, 0, 1,
                                                    0, 0, 1, 0,
                                                    0, 1, 0, 0};
  auto CNOT = qsim::Cirq::MatrixGate2<fp_type>::Create(0,
                                                       0, 1,
                                                       CNOT_mat);
  qsim::ApplyGate(simulator, CNOT,state);
  print_amplitudes(state);


  return 0;
}
