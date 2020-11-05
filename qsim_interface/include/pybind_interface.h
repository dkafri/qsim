//
// Created by dkafri on 10/30/20.
//

#ifndef QSIM_INTERFACE_INCLUDE_PYBIND_INTERFACE_H_
#define QSIM_INTERFACE_INCLUDE_PYBIND_INTERFACE_H_

#include <pybind11/numpy.h>

#include <utility>
#include "sampling.h"
#include "k_ops.h"

/** Wrapper class for a buffer object that can be wrapped by a numpy array
 * matrix.*/
template<typename fp_type>
class MatrixBuffer {
 public:
  MatrixBuffer(size_t rows, size_t cols)
      : m_rows(rows), m_cols(cols), m_data(cols * rows) {};
  fp_type* data() { return m_data.data(); }
  size_t rows() const { return m_rows; }
  size_t cols() const { return m_cols; }

  void set_value(size_t row, size_t col, fp_type val) {
    m_data.at(row * m_cols + col) = val;
  }

 private:
  size_t m_rows, m_cols;
  std::vector<fp_type> m_data;

};

/** Interface object for collecting samples
 *
 * Constructor requires num_threads and max_qubits.
 *
 * Attributes:
 * initial_state (private) - KState. Plus function to set it using input array.
 * initial_registers (private) - Setter
 * ops(private) - Function to add COperations and KOperations sequentially.
 * rng(private) - Function to set seed;
 * register_order(private) - setter
 *
 * Other methods:
 * sample_states(num_samples) - assumes initial_state/registers/register_order has already
 * been set. returns a struct storing
 *   a vector of final state vectors (c aligned)
 *   a vector of final qubit_axis (after doing c_align these should all match...)
 *   a 2d array of recorded register values, in register order.
 *
 * */

template<typename Simulator>
class Sampler {
 public:
  using fp_type = typename Simulator::fp_type;
  using StateSpace = typename Simulator::StateSpace;

  size_t max_qubits; /** Sets max required memory for representing state.*/

  /** Basic constructor */
  // We do not allocate memory for the constructed initial state since we
  // assume memory will be externally allocated.
  Sampler(size_t num_threads, size_t max_qubits)
      : num_threads(num_threads),
        max_qubits(max_qubits),
        init_kstate(num_threads, 1, std::vector<std::string>{}) {
    std::random_device rd;
    rgen = std::mt19937(rd());
  };

/** Seed the random number generator */
  void set_random_seed(size_t seed) { rgen = std::mt19937(seed); }

  void set_initial_registers(const RegisterMap& registers) {
    init_registers.clear();
    for (const auto& k_v: registers) init_registers.insert(k_v);
  };

  void set_register_order(const std::vector<std::string>& order) {
    register_order = order;
  };

  /** Set pointer to initial state vector data.
   *
   * User is responsible for allocating a complex<fp_type> vector of size
   * 2^max_qubits.
   *
   * @param array: complex numpy array storing state vector data.
   * @param axes: axis ordering for the tensor representation of the state
   *     vector. It is assumed that a qubit of memory is allocated for each
   *     axis. Axes that have dimension 1 should not be included. Axes that
   *     have dimension 2^k should be repeated k times.
   *
   * */
  using Complex = std::complex<typename Simulator::fp_type>;
  void bind_initial_state(pybind11::array_t<Complex> array,
                          const std::vector<std::string>& axes) {
    pybind11::buffer_info buffer = array.request();

    size_t expected_size = (size_t{1} << max_qubits);
    if (buffer.size != expected_size)
      throw std::runtime_error("Input array size does not match max_qubits.");

    auto ptr = static_cast<typename Simulator::fp_type*>(buffer.ptr);
    //Assign axes backwards to match qsim order
    std::vector<std::string> axes_r(axes.rbegin(), axes.rend());
    init_kstate =
        std::move(KState<Simulator>(num_threads, max_qubits, std::move(axes_r),
                                    ptr));

    auto state = init_kstate.active_state();
    //Converts from RIRIRIRI to RRRIIII
    StateSpace(num_threads).NormalToInternalOrder(state);

  }

  /** Encode and add a KOperation to the operation order.
   *
   *
   * */
  using NPCArray=pybind11::array_t<std::complex<fp_type>>;
  using StrV = std::vector<std::string>;
  typedef std::tuple<NPCArray, StrV, StrV, StrV, StrV, StrV> KOperatorData;
  using KChannelData = std::vector<KOperatorData>;
  using ChannelMapData=std::map<std::vector<size_t>, KChannelData>;

  /** Load a KOperation into the operations order.*/
  void add_koperation(ChannelMapData& channels,
                      const StrV& conditional_registers,
                      bool is_recorded,
                      const std::string& label,
                      bool is_virtual) {
    // Construct channels map
    typename KOperation<fp_type>::ChannelMap cmap;

    for (auto& regs_channels : channels) {
      // Construct the vector of KOperators (KChannel) from the data
      KChannelData& channel_data = regs_channels.second;
      KChannel<fp_type> channel_vec;
      channel_vec.reserve(channel_data.size());
      for (auto& k_op_data : channel_data)
        channel_vec.push_back(build_koperator(k_op_data));

      // Assign the KChannel to the register
      auto& reg_vals = regs_channels.first;
      cmap.emplace(reg_vals, channel_vec);
    }

    ops.emplace_back(KOperation<fp_type>(cmap, conditional_registers,
                                         is_recorded,
                                         label,
                                         is_virtual));
  }

  using COperatorData = std::tuple<COperator::TruthTable,
                                   StrV,
                                   StrV>;
  using CChannelData = std::tuple<std::vector<COperatorData>,
                                  std::vector<double>>;
  using CChannelMapData = std::map<std::vector<size_t>, CChannelData>;

  /** Load a COperation into the operations order*/
  void add_coperation(CChannelMapData& channels_data,
                      const StrV& conditional_registers,
                      std::set<std::string> added,
                      bool is_virtual) {

    //construct the channel map
    COperation::ChannelMap channel_map;

    for (auto& cond_reg_channel_data : channels_data) {
      auto& cond_reg = cond_reg_channel_data.first;
      auto& channel_data = cond_reg_channel_data.second;
      channel_map.emplace(cond_reg, build_cchannel(channel_data));
    }

    ops.emplace_back(COperation(channel_map,
                                conditional_registers, std::move(added),
                                is_virtual));

  }

  using RegisterType = uint8_t;
  /** Collect samples from simulation.*/
  MatrixBuffer<RegisterType> sample_states(size_t num_samples) {
    KState<Simulator> k_state(init_kstate);
    KState<Simulator> tmp_state(init_kstate);
    RegisterMap final_registers;

    std::set<std::string> virtual_regs;
    // Determine which virtual registers need to be recorded
    for (const auto& op : ops) {
      if (is_virtual(op)) {
        const auto& regs = saved_virtuals(op);
        virtual_regs.insert(regs.begin(), regs.end());
      }
    }

    MatrixBuffer<RegisterType> register_mat(num_samples, register_order.size());

    for (size_t ii = 0; ii < num_samples; ii++) {

      k_state.copy_from(init_kstate);
      final_registers = sample_sequence(ops,
                                        k_state,
                                        tmp_state,
                                        init_registers,
                                        rgen,
                                        virtual_regs);

      for (size_t jj = 0; jj < register_order.size(); jj++) {
        register_mat.set_value(ii,
                               jj,
                               final_registers.at(register_order.at(jj)));
      }
    }

    return register_mat;
  }
 private:
  size_t num_threads; /** Number of multi-threads for simulation.*/
  KState<Simulator> init_kstate; /** Stores initial state vector.*/
  RegisterMap init_registers; /** Initial classical registers */
  std::vector<Operation<fp_type>> ops; /** Operations to sample over.*/
  std::mt19937 rgen; /** random number generator*/
  std::vector<std::string> register_order; /** Order of all saved registers.
 * This is used to determine which virtual registers are recorded and also the
 * output array.*/

  static inline KOperator<fp_type> build_koperator(KOperatorData& data) {
    //First argument is a numpy array which we need to parse as a vector
    //This always makes a copy.
    NPCArray& matrix_arr = std::get<0>(data);
    pybind11::buffer_info buffer = matrix_arr.request();
    auto ptr = static_cast<fp_type*>(buffer.ptr);
    //Write complex array as float array with twice as many elements
    qsim::Matrix<fp_type> matrix(ptr, ptr + 2 * buffer.size);

    return KOperator<fp_type>(matrix,
                              std::get<1>(data),
                              std::get<2>(data),
                              std::get<3>(data),
                              std::get<4>(data),
                              std::get<5>(data));
  }

  static inline CChannel build_cchannel(CChannelData& data) {

    std::vector<COperatorData>& c_ops_datas = std::get<0>(data);
    std::vector<COperator> c_ops;
    c_ops.reserve(c_ops_datas.size());

    for (auto& c_ops_data : c_ops_datas)
      c_ops.push_back(COperator{std::get<0>(c_ops_data),
                                std::get<1>(c_ops_data),
                                std::get<2>(c_ops_data)});

    std::vector<double>& probs = std::get<1>(data);

    return CChannel{c_ops, probs};

  };

};

#endif //QSIM_INTERFACE_INCLUDE_PYBIND_INTERFACE_H_
