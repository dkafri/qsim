//
// Created by dkafri on 10/26/20.
//

#include <iostream>
#include <formux.h>
#include <simmux.h>
#include <sampling.h>

using namespace std;
int main() {
  using Simulator = qsim::Simulator<qsim::For>;
  using fp_type = Simulator::fp_type;
  using Operation = Operation<fp_type>;
  constexpr fp_type sqrt_half = 0.7071067811865476;

  //Simulate a simple QND measurement with readout error

  vector<string> axis_labels{"a"};
  KState<Simulator> k_state(4, 5, axis_labels);
  vector<Operation> ops;

  //Hadamard on "a"
  KOperator<fp_type> H{{sqrt_half, 0, sqrt_half, 0,
                        sqrt_half, 0, -sqrt_half, 0}, {}, {"a"}, {}, {}, {}};
  ops.emplace_back(KOperation<fp_type>::unconditioned({H}, false, "H->a"));

  // Create "b" in + state

  KOperator<fp_type>
      k_op{{sqrt_half, 0, 0, 0, sqrt_half, 0, 0, 0}, {"b"}, {"b"}, {}, {}, {}};
  ops.emplace_back(KOperation<fp_type>::unconditioned({k_op},
                                                      false,
                                                      "prep b->+"));

  //CZ between a and b
  KOperator<fp_type> CZ{{1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, -1, 0}, {}, {"a", "b"}, {}, {}, {}};
  ops.emplace_back(KOperation<fp_type>::unconditioned({CZ}, false, "CZ a->b"));

  //Swap b and c
  KOperator<fp_type> swap{{1, 0, 0, 0,
                           0, 0, 1, 0}, {}, {"b"}, {"b"}, {"c"}, {}};
  ops.emplace_back(KOperation<fp_type>::unconditioned({swap},
                                                      false,
                                                      "swap b->c"));

  // Destructively measure c in computational basis
  KOperator<fp_type> meas_0{{1, 0, 0, 0,
                             0, 0, 0, 0}, {}, {"c"}, {}, {}, {"c"}};
  KOperator<fp_type> meas_1{{0, 0, 1, 0,
                             0, 0, 0, 0}, {}, {"c"}, {}, {}, {"c"}};
  auto m_true = "measure c";
  ops.emplace_back(KOperation<fp_type>::unconditioned({meas_0, meas_1},
                                                      true,
                                                      m_true));

  //Add measurement error
  fp_type p_error = 0.1;
  string m_observe = "a_observed";
  COperator nothing{{{{0}, {0}},
                     {{1}, {1}}}, {m_true}, {m_observe}, {m_observe}};
  COperator flip{{{{0}, {1}},
                  {{1}, {0}}}, {m_true}, {m_observe}, {m_observe}};
  ops.emplace_back(COperation({{nothing, flip}, {1 - p_error, p_error}}));

  //Virtual Z on "a", then virtual measurement of "a" in X basis
  KOperator<fp_type> VZ{{1, 0, 0, 0,
                         0, 0, -1, 0}, {}, {"a"}, {}, {}, {}};
  ops.emplace_back(KOperation<fp_type>::unconditioned({VZ},
                                                      false,
                                                      "",
                                                      true));
  KOperator<fp_type> vmeas_0{{sqrt_half, 0, sqrt_half, 0,
                              0, 0, 0, 0}, {}, {"a"}, {}, {}, {"a"}};
  KOperator<fp_type> vmeas_1{{sqrt_half, 0, -sqrt_half, 0,
                              0, 0, 0, 0}, {}, {"a"}, {}, {}, {"a"}};
  string vm_label = "VM:a";
  ops.emplace_back(KOperation<fp_type>::unconditioned({vmeas_0, vmeas_1},
                                                      true,
                                                      vm_label,
                                                      true));



  //Run simulation
  random_device rd;
  std::mt19937 rgen(rd());
  KState<Simulator> tmp_state(k_state);
  auto final_registers =
      sample_sequence(ops, k_state, tmp_state, {}, rgen, {vm_label});

  cout << "Final register values:\n";
  for (const auto& k_v : final_registers)
    cout << k_v.first << ": " << k_v.second << endl;
  cout << "Final state:\n";
  k_state.print_amplitudes();

  return 0;
}
