//
// Created by dkafri on 10/28/20.
//

#include <iostream>
#include <k_ops.h>
#include <sampling.h>
#include "include/test_utils.h"
#include "../lib/simmux.h"
#include "../lib/formux.h"

#define DEBUG 0 //Set to 1 to disable acutest, which allows for debugging.

#if DEBUG
void TEST_CHECK(bool v) { assert(v); };
void TEST_CASE(const char*) {};
#else
#include "include/acutest.h"
#endif

using namespace std;
using Simulator = qsim::Simulator<qsim::For>;
using StateSpace = Simulator::StateSpace;
using fp_type= Simulator::fp_type;
using Complex = complex<fp_type>;
using Matrix = qsim::Matrix<fp_type>;

constexpr Complex one(1, 0);

void test_state_creation_destruction_case(bool swap_cnot) {
  KState<Simulator> k_state(3, 5, vector<string>{"a", "b"});

  vector<Operation<fp_type>> ops;

  //Create "c" qubit in + state
  using KOpType = KOperation<fp_type>;
  auto sqrt_half = fp_type(qsim::Cirq::is2_double);
  auto c_plus = KOperation<fp_type>(KOperator<fp_type>
                                        {{sqrt_half, 0, 0, 0, sqrt_half, 0, 0,
                                          0},
                                         {"c"}, {"c"}, {}, {}, {}});

  //CNOT between "c" and "a"
  unique_ptr<KOpType> CNOT_op_ptr;
  if (swap_cnot) {
    KOperator<fp_type> CNOT{{1, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0}, {}, {"a", "c"}, {}, {},
                            {}};
    CNOT_op_ptr = make_unique<KOpType>(KOperation<fp_type>(CNOT));
  } else {
    KOperator<fp_type> CNOT{{1, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 1, 0, 0, 0}, {}, {"c", "a"}, {}, {},
                            {}};
    CNOT_op_ptr = make_unique<KOpType>(KOperation<fp_type>(CNOT));
  }

  // Measure "c" destructively
  KOperator<fp_type> meas_0{{1, 0, 0, 0,
                             0, 0, 0, 0}, {}, {"c"}, {}, {}, {"c"}};
  KOperator<fp_type> meas_1{{0, 0, 1, 0,
                             0, 0, 0, 0}, {}, {"c"}, {}, {}, {"c"}};

  string m_label_0 = "c0";
  auto measure_c = KOperation<fp_type>({meas_0, meas_1}, true, m_label_0);
  string m_label_1 = "c1";
  auto measure_c_1 = KOperation<fp_type>({meas_0, meas_1}, true, m_label_1);

  ops.emplace_back(c_plus);
  ops.emplace_back(*CNOT_op_ptr);
  ops.emplace_back(measure_c);
  ops.emplace_back(c_plus);
  ops.emplace_back(*CNOT_op_ptr);
  ops.emplace_back(measure_c_1);


  random_device rd;
  std::mt19937 rgen(rd());
  KState<Simulator> tmp_state(k_state);
  KState<Simulator> initial_state(k_state);

  RegisterMap final_registers;
  for (size_t ii = 0; ii < 100; ii++) {
    k_state.copy_from(initial_state);
    tmp_state.copy_from(initial_state);
    final_registers = sample_sequence(ops, k_state, tmp_state, {}, rgen, {});

    // (assuming order abc:)
    // number of 1 measurements on c equals number of flips of a
    k_state.order_axes({"a", "b"});
    auto m_val_0 = final_registers.at(m_label_0);
    auto m_val_1 = final_registers.at(m_label_1);
    auto expected = (m_val_0 + m_val_1) % 2;
    auto actual = StateSpace::GetAmpl(k_state.active_state(), expected);
    TEST_CHECK(equals(actual, one));

  }
}

void test_state_creation_destruction() {

#ifdef NDEBUG
  cout << "Running tests without assertions...\n";
#else
  cout << "Running tests with assertions...\n";
#endif

  TEST_CASE("swap");
  test_state_creation_destruction_case(true);

  TEST_CASE("no_swap");
  test_state_creation_destruction_case(false);

}

void test_conditional_bit_flip() {

  KState<Simulator> k_state(5, 3, vector<string>{"a"});

  vector<Operation<fp_type>> ops;

  RegisterMap init_reg{{"X", 1}};

  //Set classical register Y to 0 or 1 with even odds
  COperator set_0{{{{}, {0}}}, {}, {"Y"}};
  COperator set_1{{{{}, {1}}}, {}, {"Y"}};

  ops.emplace_back(COperation(CChannel{{set_0, set_1}, {0.5, 0.5}}, {"Y"}));

  // Flip qubit conditioned on AND of X and Y
  KOperator<fp_type> X{{0, 0, 1, 0,
                        1, 0, 0, 0,}, {}, {"a"}, {}, {}, {}};
  KOperator<fp_type> I{{1, 0}, {}, {}, {}, {}, {}};
  KOperation<fp_type>::ChannelMap cmap{{{0, 0}, {I}},
                                       {{1, 0}, {I}},
                                       {{0, 1}, {I}},
                                       {{1, 1}, {X}}};
  ops.emplace_back(KOperation<fp_type>(cmap, {"X", "Y"}));

  //Measure a
  string m_label = "a value";
  KOperator<fp_type>
      meas_0{{1, 0, 0, 0, 0, 0, 0, 0,}, {}, {"a"}, {}, {}, {"a"}};
  KOperator<fp_type>
      meas_1{{0, 0, 1, 0, 0, 0, 0, 0,}, {}, {"a"}, {}, {}, {"a"}};
  ops.emplace_back(KOperation<fp_type>({meas_0, meas_1}, true, m_label));

  random_device rd;
  std::mt19937 rgen(rd());
  KState<Simulator> tmp_state(k_state);
  KState<Simulator> initial_state(k_state);

  RegisterMap final_registers;
  for (size_t ii = 0; ii < 100; ii++) {
    k_state.copy_from(initial_state);
    final_registers =
        sample_sequence(ops, k_state, tmp_state, init_reg, rgen, {});

    auto m_val = final_registers.at(m_label);
    auto X_val = final_registers.at("X");
    auto Y_val = final_registers.at("Y");
    TEST_CHECK(equals(m_val, X_val & Y_val));

  }

}

void test_virtual_operations_no_effect() {

  vector<string> axes{"a", "b"};
  KState<Simulator> k_state(2, 4, axes);

  vector<Operation<fp_type>> ops;

  //Virtual bit flip on "a"
  KOperator<fp_type> XV{{0, 0, 1, 0,
                         1, 0, 0, 0}, {}, {"a"}, {}, {}, {}};
  ops.emplace_back(KOperation<fp_type>(XV, false, "XV", true));

  //virtual measurement of "a"
  KOperator<fp_type> meas_0{{1, 0, 0, 0,
                             0, 0, 0, 0}, {}, {"a"}, {}, {}, {"a"}};
  KOperator<fp_type> meas_1{{0, 0, 1, 0,
                             0, 0, 0, 0}, {}, {"a"}, {}, {}, {"a"}};
  string vm_label = "a (V)";
  ops.emplace_back(KOperation<fp_type>({meas_0, meas_1,},
                                       true,
                                       vm_label,
                                       true));

  //virtual copy of a into a new register
  string vm_label2 = "b (V)";
  COperator cp{{{{0}, {1}},
                {{1}, {1}}}, {vm_label}, {vm_label2}};
  ops.emplace_back(COperation{cp, {vm_label2}, true});

  random_device rd;
  std::mt19937 rgen(rd());
  KState<Simulator> tmp_state(k_state);

  RegisterMap final_registers;
  final_registers =
      sample_sequence(ops, k_state, tmp_state, {}, rgen, {vm_label, vm_label2});

  //virtual registers recorded
  TEST_CHECK(final_registers.at(vm_label) == 1);
  TEST_CHECK(final_registers.at(vm_label2) == 1);
  // but "a" was not destroyed or flipped.
  TEST_CHECK(k_state.qubit_of("a") == 0);

  k_state.order_axes({"b", "a"});
  auto state = k_state.active_state();
  TEST_CHECK(Simulator::StateSpace::GetAmpl(state, 0) == Complex(1, 0));

}

#if DEBUG
int main() {
  test_state_creation_destruction_case(true);
  return 0;
}
#else
TEST_LIST = {
    {"create destroy", test_state_creation_destruction},
//    {"conditional ops", test_conditional_bit_flip},
//    {"virtual operations no effect", test_virtual_operations_no_effect},
    {nullptr, nullptr} // Required final element
};
#endif






