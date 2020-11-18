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

  /** assert that data has required structure.*/
  void validate() const {
    for (const auto& key_value : data) {
      if (key_value.first.size() != inputs.size()) {
        std::stringstream ss;
        ss << "COperation: Tuple of input register values (";
        for (const auto& val: key_value.first)
          ss << val << ",";
        ss << ") does is inconsistent with the length of inputs (";
        for (const auto& reg : inputs)
          ss << reg << ",";
        ss << ").";
        throw std::runtime_error(ss.str());
      }
      if (key_value.second.size() != outputs.size()) {
        std::stringstream ss;
        ss << "COperation: Tuple of output register values (";
        for (const auto& val: key_value.second)
          ss << val << ",";
        ss << ") does is inconsistent with the length of outputs (";
        for (const auto& reg : outputs)
          ss << reg << ",";
        ss << ").";
        throw std::runtime_error(ss.str());
      }

    }

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

    for (auto&& reg  :inputs) {
      assert(registers.count(reg));
      input_vec.push_back(registers.at(reg));
    }

    //Write output values to register
#ifndef NDEBUG
    if (!data.count(input_vec)) {
      std::cerr << "No output defined for register values: (";
      for (const auto& val: input_vec)
        std::cerr << val << ",";
      std::cerr << "), for expected inputs (";
      for (const auto& reg : inputs)
        std::cerr << reg << ",";
      std::cerr << "),\nand outputs (";
      for (const auto& reg : outputs)
        std::cerr << reg << ",";
      std::cerr << ")\n";
    }
    assert(data.count(input_vec));

#endif
    const std::vector<size_t>& output_vec = data.at(input_vec);
    auto val_ptr = output_vec.begin();
    for (auto reg_ptr = outputs.begin(); reg_ptr != outputs.end();
         ++reg_ptr, ++val_ptr)
      registers[*reg_ptr] = *val_ptr;
  }
};

/** Representation of a stochastic operation on classical registers.*/
struct CChannel {

  std::vector<COperator> operators; /** Operators to apply probabilistically.*/
  std::vector<double> probs; /** Probability of each table. Elements must be
 * non-negative and sum to 1.*/

  /** Assert that data satisfies requirements.*/
  void validate(const std::set<std::string>& added) const {

    if (operators.size() != probs.size()) {
      std::stringstream ss;
      ss << "Number of operators for CChannel (" << operators.size() << ") ";
      ss << "does not equal number of probabilities (" << probs.size() << ").";
      throw std::runtime_error(ss.str());
    }
    if (operators.empty())
      return;

    double sum = 0.0;
    for (const auto& prob : probs) {
      if (0 > prob || prob > 1)
        throw std::runtime_error("COperator probability not between 0 and 1.");
      sum += prob;
    }

    if (fabs(1.0 - sum) > 1e-12) {
      std::stringstream ss;
      ss << "Sum of COperator probabilities " << sum << " is not close to 1.";
      throw std::runtime_error(ss.str());
    }

    for (const auto& op:operators) {
      op.validate();
      for (const auto& reg: added) // Check each element of added is in outputs.
        if (std::find(op.outputs.begin(), op.outputs.end(), reg)
            == op.outputs.end()) {
          throw std::runtime_error(
              "Added axis ( " + reg + " ) not present in classical operator "
                                      "outputs.");
        }
    }

  }

  /** Apply this operation to a set of classical registers.*/
  void apply(std::unordered_map<std::string, size_t>& registers,
             double cutoff) const {
    if (operators.empty()) //Trivial case, do nothing
      return;

    //Sample operator to apply
    assert(0 <= cutoff && cutoff < 1);
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
  std::set<std::string> added; /** Output registers that are created by
 * this operation. This must be consistent for all channels.*/
  bool is_virtual; /** Whether the operation is virtual. Virtual
 * operations have a temporary effect on the simulation and are back-tracked
 * when a non-virtual operation is met.*/

  /** Assert that data satisfies requirements.*/
  void validate() const {
    for (const auto& key_val : channels) {
      assert(key_val.first.size() == conditional_registers.size());
      key_val.second.validate(added);
    }
  }

  /** Constructor for a single deterministic operation.*/
  //If I instead initialized in the function body, we would get errors. Why?
  explicit COperation(const COperator& op,
                      std::set<std::string> added,
                      bool is_virtual = false)
      : is_virtual(is_virtual),
        channels{{{}, CChannel{{op}, {1.0}}}},
        added(std::move(added)) {
    validate();
  }

  /** Constructor for an unconditioned stochastic operation.*/
  explicit COperation(const CChannel& channel,
                      std::set<std::string> added,
                      bool is_virtual = false)
      : is_virtual(is_virtual), channels{{{}, channel}},
        added(std::move(added)) { validate(); }

  /** Generic constructor */
  COperation(ChannelMap cmap,
             std::vector<std::string> conditional_registers,
             std::set<std::string> added,
             bool is_virtual = false)
      : channels(std::move(cmap)),
        conditional_registers(std::move(conditional_registers)),
        is_virtual(is_virtual),
        added(std::move(added)) { validate(); }

  /** Apply this operation to a set of classical registers.*/
  void apply(std::unordered_map<std::string, size_t>& registers,
             double cutoff) const {
    //Trivial case, do nothing
    if (channels.empty())
      return;


    //Read conditional registers
    std::vector<size_t> cond_vec;
    cond_vec.reserve(conditional_registers.size());

    for (const auto& reg : conditional_registers) {
      assert(registers.count(reg));
      cond_vec.push_back(registers.at(reg));
    }



    // Apply appropriate channel
    assert(channels.count(cond_vec));
    const auto& channel = channels.at(cond_vec);
#ifdef NDEBUG
    for (const auto & reg : added){
      assert(registers.count(reg));
    }
#endif
    channel.apply(registers, cutoff);
  }

};

#endif //QSIM_INTERFACE_INCLUDE_C_OPS_H_
