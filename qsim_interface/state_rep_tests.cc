//
// Created by dkafri on 10/13/20.
//

//
// Created by dkafri on 3/27/20.
//

#include <iostream>
#include <cassert>
#include "include/test_utils.h"
#include "../lib/simmux.h"
#include "../lib/formux.h"
#include "include/state_rep.h"

#define DEBUG 0 //Set to 1 to disable acutest, which allows for debugging.

#if DEBUG
void TEST_CHECK(bool v) { assert(v); };
void TEST_CASE(const char*) {};
#else
#include "include/acutest.h"
#endif

using namespace std;
using Simulator = qsim::Simulator<qsim::For>;
using Complex = complex<Simulator::fp_type>;
using Matrix= qsim::Matrix<Simulator::fp_type>;

void test_kstate_space_multiply_norm() {
  vector<string> axis_labels = {"a", "b"};
  KState<Simulator> k_state(3, 4, axis_labels);

  auto space = k_state.active_state_space();

  TEST_CHECK(almost_equals(space.Norm(k_state.active_state()), 1.0));

  auto state = k_state.active_state();
  space.Multiply(2, state);

  TEST_CHECK(almost_equals(space.Norm(k_state.active_state()), 4.0));
}

void test_kstate_add_remove_qubits_size_grows() {
  KState<Simulator> k_state(3, 5, vector<string>{});

  unsigned expected = 0;
  auto state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

  k_state.add_qubit("a");
  expected += 1;
  state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

  k_state.add_qubit("a");
  expected += 1;
  state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

  k_state.add_qubit("b");
  expected += 1;
  state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

  k_state.add_qubit("c");
  expected += 1;
  state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

  k_state.remove_qubit("b");
  expected -= 1;
  state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

  k_state.remove_qubit("a");
  expected -= 1;
  state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

  k_state.remove_qubit("c");
  expected -= 1;
  state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

}

void test_kstate_correct_qubit_removed() {
  KState<Simulator> k_state(3, 5, vector<string>{"c", "b", "b"});
  using StateSpace = KState<Simulator>::StateSpace;
  using fp_type = Simulator::fp_type;

  auto b_qubits = k_state.qubits_of("b");
  TEST_CHECK(equals(b_qubits, vector<unsigned>{1, 2}));


  // Last added qubit is slowest to change when running through states.
  // Prepare state |100> by applying X to qubit 2
  // (first qubit in standard order).
  auto state = k_state.active_state();
  auto simulator = k_state.active_simulator();
  auto X = qsim::Cirq::X<Simulator::fp_type>::Create(0, 0).matrix;
  simulator.ApplyGate(vector<unsigned>{2}, X.data(), state);
  auto actual = StateSpace::GetAmpl(state, 4);
  TEST_CHECK(equals(actual, complex<fp_type>{1}));


  // remove c qubit. This should swap the first and third qubits so now we will
  // have amplitude in |01>.
  k_state.remove_qubit("c");
  state = k_state.active_state();
  actual = StateSpace::GetAmpl(state, 1);
  TEST_CHECK(equals(actual, complex<fp_type>{1}));

}

void test_kstate_copy_constructor() {
  KState<Simulator> k_state(3, 5, vector<string>{});

  k_state.add_qubit("a");
  k_state.add_qubit("a");
  k_state.add_qubit("b");
  k_state.add_qubit("c");

  auto state = k_state.active_state();
  KState<Simulator>::StateSpace::SetAmpl(state, 2, 0.2, 3.7);

  KState<Simulator> copy(k_state);

  auto state_copy = copy.active_state();
  for (size_t i = 0; i < 16; i++) {
    auto actual = KState<Simulator>::StateSpace::GetAmpl(state_copy, i);
    auto expected = KState<Simulator>::StateSpace::GetAmpl(state, i);
    TEST_CHECK(actual == expected);
  }
  TEST_CHECK(state_copy.num_qubits() == state.num_qubits());
}

void test_remove_qubits_of() {

  KState<Simulator> k_state(3, 5, vector<string>{"c", "a", "b", "a", "b"});

  k_state.remove_qubits_of({"a", "b", "c"});

  for (const auto& ax : {"a", "b", "c"})
    TEST_CHECK(k_state.qubits_of(ax).empty());

}



void test_apply_matrix() {
  KState<Simulator> k_state(3, 5, vector<string>{"c", "b"});

  //X on C
  Matrix X{0, 0, 1, 0, 1, 0, 0, 0};
  vector<string> axes{"c"};
  k_state.permute_and_apply(X, axes);
  //State |10> (|01> in reverse order)
  Complex one{1};
  auto state = k_state.active_state();
  TEST_CHECK(equals(Simulator::StateSpace::GetAmpl(state, 1), one));
  //CNOT
  qsim::Matrix<Simulator::fp_type> matrix{1, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 1, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 1, 0,
                                          0, 0, 0, 0, 1, 0, 0, 0};

  axes = {"c", "b"};
  k_state.permute_and_apply(matrix, axes);
  TEST_CHECK(equals(axes, vector<string>{"b", "c"}));

  state = k_state.active_state();
  // |10> CNOT-> |11>
  TEST_CHECK(equals(Simulator::StateSpace::GetAmpl(state, 3), one));

  matrix = {1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 1, 0, 0, 0}; // undo permutation, but axes has changed
  axes = {"b", "c"};

  k_state.permute_and_apply(matrix, axes);
  // Axis now matches reverse qubit order, expect no change in axes
  TEST_CHECK(equals(axes, vector<string>{"b", "c"}));


  // |11> CNOT-> |01> (|10> in reverse order)
  state = k_state.active_state();
  TEST_CHECK(equals(Simulator::StateSpace::GetAmpl(state, 2), one));

}

void test_kstate_copy_from() {
  KState<Simulator> k_state(3, 5, vector<string>{});

  k_state.add_qubit("a");
  k_state.add_qubit("a");
  k_state.add_qubit("b");
  k_state.add_qubit("c");

  auto state = k_state.active_state();
  KState<Simulator>::StateSpace::SetAmpl(state, 2, 0.2, 3.7);

  KState<Simulator> copy(3, 5, vector<string>{});
  copy.copy_from(k_state);

  auto state_copy = copy.active_state();
  for (size_t i = 0; i < 16; i++) {
    auto actual = KState<Simulator>::StateSpace::GetAmpl(state_copy, i);
    auto expected = KState<Simulator>::StateSpace::GetAmpl(state, i);
    TEST_CHECK(actual == expected);
  }
  TEST_CHECK(state_copy.num_qubits() == state.num_qubits());
}

void test_kstate_apply_1q_gate() {
  KState<Simulator> k_state(3, 5, vector<string>{"a", "b", "c"});
  using StateSpace = KState<Simulator>::StateSpace;


  // Start in 000 state
  auto state = k_state.active_state();
  TEST_CHECK(equals(StateSpace::GetAmpl(state, 0), Complex{1}));

  Matrix X_mat{0, 0, 1, 0, 1, 0, 0, 0};
  vector<string> axes{"a"};
  k_state.permute_and_apply(X_mat, axes);
  vector<unsigned> expected{0};
  TEST_CHECK(equals(k_state.qubits_of("a"), expected));

  //first qubit index changes fastest, in state |0,0,1>
  TEST_CHECK(equals(StateSpace::GetAmpl(state, 1), Complex{1}));

  axes = {"b"};
  k_state.permute_and_apply(X_mat, axes);
  expected[0] = 1;
  TEST_CHECK(equals(k_state.qubits_of("b"), expected));
  //State |011>
  TEST_CHECK(equals(StateSpace::GetAmpl(state, 3), Complex{1}));

  axes = {"c"};
  k_state.permute_and_apply(X_mat, axes);
  expected[0] = 2;
  TEST_CHECK(equals(k_state.qubits_of("c"), expected));
  //State |111>
  TEST_CHECK(equals(StateSpace::GetAmpl(state, 7), Complex{1}));

}

void test_kstate_apply_2q_gate() {
  KState<Simulator> k_state(3, 5, vector<string>{"a", "b", "c"});
  using StateSpace = KState<Simulator>::StateSpace;


  // Start in 000 state
  auto state = k_state.active_state();
  const Complex& one = Complex{1};
  TEST_CHECK(equals(StateSpace::GetAmpl(state, 0), one));

  //Apply X on "a", which has qubit 0
  Matrix X_mat{0, 0, 1, 0, 1, 0, 0, 0};
  vector<string> axes{"a"};
  k_state.permute_and_apply(X_mat, axes);
  vector<unsigned> expected{0};
  TEST_CHECK(equals(k_state.qubits_of("a"), expected));

  //first qubit index changes fastest, in state |0,0,1>
  TEST_CHECK(equals(StateSpace::GetAmpl(state, 1), one));

  //CNOT from "a" to "b"
  Matrix CNOT_mat{1, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 1, 0, 0, 0};
  axes = {"a", "b"};

  k_state.permute_and_apply(CNOT_mat, axes);
  //State |011>
  TEST_CHECK(equals(StateSpace::GetAmpl(state, 3), one));

  //CNOT from "c" to "a" (does nothing)
  CNOT_mat = {1, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 1, 0, 0, 0};
  axes = {"c", "a"};
  k_state.permute_and_apply(CNOT_mat, axes);
  TEST_CHECK(equals(StateSpace::GetAmpl(state, 3), one));



  //CNOT from "b" to "c" --> |111>
  CNOT_mat = {1, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 1, 0, 0, 0};
  axes = {"b", "c"};
  k_state.permute_and_apply(CNOT_mat, axes);
  TEST_CHECK(equals(StateSpace::GetAmpl(state, 7), one));

}

void test_kstate_transfer_qubits() {
  KState<Simulator> k_state(3, 5, vector<string>{"a", "a"});
  using StateSpace = KState<Simulator>::StateSpace;

  auto a_qubits = k_state.qubits_of("a");
  k_state.transfer_qubits("a", "b");
  auto b_qubits = k_state.qubits_of("b");
  TEST_CHECK(equals(a_qubits, b_qubits));

}

#if DEBUG
int main() {

  test_remove_qubits_of();
  return 0;
}
#else
TEST_LIST = {
    {"norm_square_and_multiply", test_kstate_space_multiply_norm},
    {"add_remove_qubit_state_size", test_kstate_add_remove_qubits_size_grows},
    {"copy_constructor", test_kstate_copy_constructor},
    {"copy_from", test_kstate_copy_from},
    {"qubit_remove_order", test_kstate_correct_qubit_removed},
    {"apply sq gate", test_kstate_apply_1q_gate},
    {"apply 2q gate", test_kstate_apply_2q_gate},
    {"transfer qubits", test_kstate_transfer_qubits},
    {"remove multiple qubits", test_remove_qubits_of},
    {"apply matrix", test_apply_matrix},
    {nullptr, nullptr} // Required final element
};
#endif




