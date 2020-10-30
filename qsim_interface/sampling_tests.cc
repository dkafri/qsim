//
// Created by dkafri on 10/28/20.
//

#include <iostream>
#include <cassert>
#include <k_ops.h>
#include <sampling.h>
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
using StateSpace = Simulator::StateSpace;
using fp_type= Simulator::fp_type;
using Complex = complex<fp_type>;
using Matrix = qsim::Matrix<fp_type>;

constexpr Complex one(1, 0);

void test_state_creation_destruction() {

  KState<Simulator> k_state(3, 5, vector<string>{"a", "b"});

  vector<Operation<fp_type>> ops;

  //Create "c" qubit in + state
  auto sqrt_half = fp_type(qsim::Cirq::is2_double);
  KOperator<fp_type> c_plus{{sqrt_half, 0, 0, 0, sqrt_half, 0, 0, 0},
                            {"c"}, {"c"}, {}, {}, {}};
  ops.emplace_back(KOperation<fp_type>(c_plus));

  //CNOT between "c" and "a"
  KOperator<fp_type> CNOT{{1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 1, 0, 0, 0}, {}, {"c", "a"}, {}, {}, {}};
  ops.emplace_back(KOperation<fp_type>(CNOT));

  // Measure "a" destructively
  KOperator<fp_type> meas_0{{1, 0, 0, 0,
                             0, 0, 0, 0}, {}, {"a"}, {}, {}, {"a"}};
  KOperator<fp_type> meas_1{{0, 0, 1, 0,
                             0, 0, 0, 0}, {}, {"a"}, {}, {}, {"a"}};
  string m_label = "a";
  ops.emplace_back(KOperation<fp_type>({meas_0, meas_1}, true, m_label));

  random_device rd;
  std::mt19937 rgen(rd());
  KState<Simulator> tmp_state(k_state);
  KState<Simulator> initial_state(k_state);

  RegisterMap final_registers;
  for (size_t ii = 0; ii < 100; ii++) {
    k_state.copy_from(initial_state);
    final_registers = sample_sequence(ops, k_state, tmp_state, {}, rgen, {});

    k_state.c_align(); // remaining axes are "bc", so 0 and 1 are |00> and |01>
    auto m_val = final_registers.at(m_label); // a measurement maps to c state
    auto actual = StateSpace::GetAmpl(k_state.active_state(), m_val);
    TEST_CHECK(equals(actual, one));

  }
}

void test_conditional_bit_flip() {

  KState<Simulator> k_state(5, 3, vector<string>{"a"});

  vector<Operation<fp_type>> ops;

  RegisterMap init_reg{{"X", 1}};

  //Set classical register Y to 0 or 1 with even odds
  COperator set_0{{{{}, {0}}}, {}, {"Y"}, {"Y"}};
  COperator set_1{{{{}, {1}}}, {}, {"Y"}, {"Y"}};

  ops.emplace_back(COperation(CChannel{{set_0, set_1}, {0.5, 0.5}}));

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

#if DEBUG
int main() {

  return 0;
}
#else
TEST_LIST = {
    {"create destroy", test_state_creation_destruction},
    {"conditional ops", test_conditional_bit_flip},
    {nullptr, nullptr} // Required final element
};
#endif






