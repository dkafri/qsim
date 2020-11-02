//
// Created by dkafri on 10/23/20.
//

#ifndef QSIM_INTERFACE_INCLUDE_C_OPS_H_
#define QSIM_INTERFACE_INCLUDE_C_OPS_H_

#include <utility>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <cassert>
#include <algorithm>
#include <cmath>
/** Representation of deterministic operation on classical registers.*/
struct COperator {

  using TruthTable = std::map<std::vector<size_t>, std::vector<size_t>>;
  TruthTable data; /** Map storing all
 * expected input-output combinations.*/
  std::vector<std::string> inputs; /** Input registers. Must have same length
 * as all keys of data.*/
  std::vector<std::string> outputs; /** Outputs registers. Must have same length
 * as all values of data. If not outputs are specified then this operation
 * deletes all input registers.*/
  std::set<std::string> added; /** Output registers that are created by
 * this operation. Each element must be a subset of outputs.*/

  /** assert that data has required structure.*/
  void validate() const {
    for (const auto& key_value : data) {
      assert(key_value.first.size() == inputs.size());
      assert(key_value.second.size() == outputs.size());
    }
    for (const auto& reg: added) // Check each element of added is in outputs.
      assert(std::find(outputs.begin(), outputs.end(), reg) != outputs.end());
  }

  /** Apply this operator to a set of classical registers.*/
  void apply(std::unordered_map<std::string, size_t>& registers) const {
    //Special case for register deletion or null case
    if (outputs.empty()) {
      for (auto&& reg: inputs)
        registers.erase(reg);
      return;
    }

    //Determine inputs
    std::vector<size_t> input_vec;
    input_vec.reserve(inputs.size());

    for (auto&& reg  :inputs)
      input_vec.push_back(registers.at(reg));

    //Add registers
    for (auto&& reg: added)
      registers.emplace(reg, 0);

    //Write output values to register
    const std::vector<size_t>& output_vec = data.at(input_vec);
    auto val_ptr = output_vec.begin();
    for (auto reg_ptr = outputs.begin(); reg_ptr != outputs.end();
         ++reg_ptr, ++val_ptr)
      registers.at(*reg_ptr) = *val_ptr;
  }
};

/** Representation of a stochastic operation on classical registers.*/
struct CChannel {

  std::vector<COperator> operators; /** Operators to apply probabilistically.*/
  std::vector<double> probs; /** Probability of each table. Elements must be
 * non-negative and sum to 1.*/

  /** Assert that data satisfies requirements.*/
  void validate() const {
    assert(operators.size() == probs.size());
    if (operators.empty())
      return;

    double sum = 0.0;
    for (const auto& prob : probs) {
      assert(0 <= prob && prob <= 1);
      sum += prob;
    }

    assert(fabs(1.0 - sum) < 1e-12);
    for (const auto& op:operators)
      op.validate();
  }

  /** Apply this operation to a set of classical registers.*/
  void apply(std::unordered_map<std::string, size_t>& registers,
             double cutoff) const {
    if (operators.empty()) //Trivial case, do nothing
      return;

    //Sample operator to apply
    auto cop_ptr = operators.begin();
    for (const auto& prob : probs) {
      cutoff -= prob;
      if (cutoff <= 0)
        break;
      ++cop_ptr;
    }
    (*cop_ptr).apply(registers);
  }

};

/** Representation of a conditional a stochastic operation on classical
 * registers.*/
struct COperation {

  using ChannelMap=std::map<std::vector<size_t>, CChannel>;
  ChannelMap channels; /** Mapping between conditional register values and
 * corresponding channels.*/
  std::vector<std::string> conditional_registers; /**
 * Classical registers on which the operation is conditioned. Each key of
 * channels should have the same length as conditional_registers. The
 * register values are referenced in the specified order.*/
  bool is_virtual; /** Whether the operation is virtual. Virtual
 * operations have a temporary effect on the simulation and are back-tracked
 * when a non-virtual operation is met.*/

  /** Assert that data satisfies requirements.*/
  void validate() const {
    for (const auto& key_val : channels) {
      assert(key_val.first.size() == conditional_registers.size());
      key_val.second.validate();
    }
  }

  /** Constructor for a single deterministic operation.*/
  //If I instead initialized in the function body, we would get errors. Why?
  explicit COperation(const COperator& op, bool is_virtual = false)
      : is_virtual(is_virtual), channels{{{}, CChannel{{op}, {1.0}}}} {
    validate();
  }

  /** Constructor for an unconditioned stochastic operation.*/
  explicit COperation(const CChannel& channel, bool is_virtual = false)
      : is_virtual(is_virtual), channels{{{}, channel}} { validate(); }

  /** Generic constructor */
  COperation(ChannelMap cmap,
             std::vector<std::string> conditional_registers,
             bool is_virtual = false)
      : channels(std::move(cmap)),
        conditional_registers(std::move(conditional_registers)),
        is_virtual(is_virtual) { validate(); }

  /** Apply this operation to a set of classical registers.*/
  void apply(std::unordered_map<std::string, size_t>& registers,
             double cutoff) const {
    //Trivial case, do nothing
    if (channels.empty())
      return;


    //Read conditional registers
    std::vector<size_t> cond_vec;
    cond_vec.reserve(conditional_registers.size());

    for (const auto& reg : conditional_registers)
      cond_vec.push_back(registers.at(reg));


    // Apply appropriate channel
    const auto& channel = channels.at(cond_vec);
    channel.apply(registers, cutoff);
  }

};

#endif //QSIM_INTERFACE_INCLUDE_C_OPS_H_
