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
using fp_type = Simulator::fp_type;
using Matrix= qsim::Matrix<Simulator::fp_type>;

void test_kstate_space_rescale_norm() {
  vector<string> axis_labels = {"a", "b"};
  KState<Simulator> k_state(3, 4, axis_labels);

  TEST_CHECK(almost_equals(k_state.norm_squared(), 1.0));

  k_state.rescale(2.0);

  TEST_CHECK(almost_equals(k_state.norm_squared(), 4.0));
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

  k_state.remove_qubits_of({"b"});
  expected -= 1;
  state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

  k_state.remove_qubits_of({"a"});
  expected -= 2;
  state = k_state.active_state();
  TEST_CHECK(equals(expected, state.num_qubits()));

  k_state.remove_qubits_of({"c"});
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

  qsim::Matrix<fp_type> X{0, 0, 1, 0, 1, 0, 0, 0};
  // b has 2 qubits, (1, and 2). The matrix is applied in reverse order with
  // respect to most recently assigned qubits.
  vector<string> axes{"b"};
  k_state.permute_and_apply(X, axes);
  auto actual = StateSpace::GetAmpl(state, 4);
  TEST_CHECK(equals(actual, complex<fp_type>{1}));


  // remove c qubit. This should swap the first and third qubits so now we will
  // have amplitude in |01>.
  k_state.remove_qubits_of({"c"});
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

void test_sort_axes_axis_order() {

  auto unique_axes = vector<string>{"a", "b", "c"};
  auto qubit_axis = vector<string>{"c", "a", "b", "a", "b"};
  KState<Simulator> k_state(4, 5, qubit_axis);

  TEST_CHECK(equals(k_state.qubits_of("a"), vector<unsigned>{1, 3}));
  TEST_CHECK(equals(k_state.qubits_of("b"), vector<unsigned>{2, 4}));
  TEST_CHECK(equals(k_state.qubits_of("c"), vector<unsigned>{0}));

  k_state.c_align();

  TEST_CHECK(equals(k_state.qubits_of("a"), vector<unsigned>{0, 1}));
  TEST_CHECK(equals(k_state.qubits_of("b"), vector<unsigned>{2, 3}));
  TEST_CHECK(equals(k_state.qubits_of("c"), vector<unsigned>{4}));
}

inline size_t get_bit(size_t num, size_t which_bit) {
  constexpr size_t one(1);
  return (num & (one << which_bit)) >> which_bit;
}

void test_sort_axes_correct_state(size_t init_state) {
  TEST_CHECK(0 <= init_state);

  auto qubit_axis = vector<string>{"c", "a", "b"};
  const size_t num_axes = qubit_axis.size();
  TEST_CHECK(init_state < size_t{1} << num_axes);
  auto ordered_axis(qubit_axis);
  std::sort(ordered_axis.begin(), ordered_axis.end());

  KState<Simulator> k_state(4, 5, qubit_axis);

  //Set system to initial state in expected alphabetical order
  Matrix X{0, 0, 1, 0,
           1, 0, 0, 0};
  vector<string> axes;

  //Apply bit flips on axes to prepare expected state
  for (size_t bit_ind = 0; bit_ind < num_axes; bit_ind++) {
    if (get_bit(init_state, bit_ind)) {
      axes = {ordered_axis[num_axes - bit_ind - 1]};
      k_state.permute_and_apply(X, axes);
    }
  }

  k_state.c_align();

  auto state = k_state.active_state();
  auto actual = Simulator::StateSpace::GetAmpl(state, init_state);
  TEST_CHECK(actual == Complex(1));

}

void test_sort_axes_correct_state() {

  for (size_t init_state = 0; init_state < 8; init_state++) {
    TEST_CASE(to_string(init_state).c_str());
    test_sort_axes_correct_state(init_state);

  }

}

void test_f_align_reverse_c_align() {

  KState<Simulator> k_state(4, 5, vector<string>{"d", "d", "a", "b", "c"});

  //X on "b", CNOT from b to a
  Matrix X{0, 0, 1, 0,
           1, 0, 0, 0};
  vector<string> axes{"a"};
  k_state.permute_and_apply(X, axes);

  k_state.c_align();
  auto state = k_state.active_state();
  //|10000> -> 16
  TEST_CHECK(Simulator::StateSpace::GetAmpl(state, 16) == Complex{1});

  k_state.f_align(); // Now we can apply matrices again.

  Matrix CNOT{1, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 1, 0, 0, 0};
  axes = {"a", "b"};
  k_state.permute_and_apply(CNOT, axes);

  k_state.c_align();
  //|11000> -> 16+8=24
  TEST_CHECK(Simulator::StateSpace::GetAmpl(state, 24) == Complex{1});

}

void test_pointer_constructor() {

  size_t max_qubits = 3;
  size_t size =
      2 * (size_t{1} << max_qubits); //data stored as R,R,R,...,I,I,I,....
  fp_type* data;
  data = new fp_type[size];

  for (size_t ii = 0; ii < size / 2; ii++) {
    data[ii] = fp_type(2 * ii);
    data[ii + size / 2] = fp_type(2 * ii + 1);
  }

  for (;;) {
    KState<Simulator>
        k_state(3, max_qubits, vector<string>{"a", "b", "c"}, data);

    auto state = k_state.active_state();
    //confirm data storage convention.
    for (size_t ii = 0; ii < size / 2; ii++) {
      auto actual = Simulator::StateSpace::GetAmpl(state, ii);
      Complex expected(ii * 2, 2 * ii + 1);
      TEST_CHECK(actual == expected);
    }
    break;
  }

  // removing KState from scope does not delete data
  for (size_t ii = 0; ii < size / 2; ii++) {
    TEST_CHECK(data[ii] == fp_type(2 * ii));
    TEST_CHECK(data[ii + size / 2] == fp_type(2 * ii + 1));
  }

  delete[]data;

}

#if DEBUG
int main() {

  test_sort_axes_axis_order();
  return 0;
}
#else
TEST_LIST = {
    {"norm_square_and_multiply", test_kstate_space_rescale_norm},
    {"add_remove_qubit_state_size", test_kstate_add_remove_qubits_size_grows},
    {"copy_constructor", test_kstate_copy_constructor},
    {"copy_from", test_kstate_copy_from},
    {"qubit_remove_order", test_kstate_correct_qubit_removed},
    {"apply sq gate", test_kstate_apply_1q_gate},
    {"apply 2q gate", test_kstate_apply_2q_gate},
    {"transfer qubits", test_kstate_transfer_qubits},
    {"remove multiple qubits", test_remove_qubits_of},
    {"apply matrix", test_apply_matrix},
    {"sort axes", test_sort_axes_axis_order},
    {"sort axes state preserved", test_sort_axes_correct_state},
    {"f_align reverses c_align", test_f_align_reverse_c_align},
    {"pointer constructor", test_pointer_constructor},
    {nullptr, nullptr} // Required final element
};
#endif




