//
// Created by dkafri on 10/30/20.
//

#ifndef QSIM_INTERFACE_INCLUDE_QSIM_KRAUS_SIM_H_
#define QSIM_INTERFACE_INCLUDE_QSIM_KRAUS_SIM_H_

#include <pybind11/numpy.h>

#include <utility>
#include "sampling.h"
#include "k_ops.h"

template<typename fp_type>
pybind11::array_t<fp_type,
                  pybind11::array_t<fp_type>::c_style> as_pyarray(
    std::vector<size_t> shape, const fp_type* data) {
  //Encapsulate the data in a pybind array object. First use a capsule
  // wrapper to ensure the object is not garbage collected immediately.
  auto capsule = pybind11::capsule(
      data, [](void* data) { delete[] reinterpret_cast<fp_type*>(data); });

  return pybind11::array_t<fp_type>(shape, data, capsule);
}

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
  using State = typename StateSpace::State;

  bool consistent_axis_order; /** Whether every sample is expected to always
 * have the same qubit axis order. This is True when there all KrausOperations
 * have consistent output shapes.*/

  /** Basic constructor */
  // We do not allocate memory for the constructed initial state since we
  // assume memory will be externally allocated.
  Sampler(size_t num_threads, bool consistent_axis_order)
      : consistent_axis_order(consistent_axis_order), num_threads(num_threads),
        init_kstate(num_threads, 1, std::vector<std::string>{}) {
    std::random_device rd;
    rgen = std::mt19937(rd());
  };

/** Seed the random number generator */
  void set_random_seed(size_t seed) { rgen = std::mt19937(seed); }

/** Specify classical registers that exist at the start of the simulation.*/
  void set_initial_registers(const RegisterMap& registers) {
    init_registers.clear();
    for (const auto& k_v: registers) init_registers.insert(k_v);
  };

/** Specify order of registers that are to be returned at the end of a
 *  a simulation. Registers not included in this sequence are not recorded.*/
  void set_register_order(const std::vector<std::string>& order) {
    register_order = order;
  };

  /** Set pointer to initial state vector data.
   *
   * User is responsible for allocating a complex<fp_type> vector of size
   * 2^num_qubits (we assume each qubit has 2 levels).
   *
   * @param array: complex numpy array storing state vector data.
   * @param axes: axis ordering for the tensor representation of the state
   *     vector. It is assumed that a qubit of memory is allocated for each
   *     axis. Axes that have dimension 1 should not be included. Axes that
   *     have dimension 2^k should be repeated k times.
   *
   * */
  using Complex = std::complex<typename Simulator::fp_type>;
  void bind_initial_state(pybind11::array_t<Complex,
                                            pybind11::array::c_style> array,
                          const std::vector<std::string>& axes) {

    size_t array_size = array.size();
    if ((array_size & (array_size - 1)) != 0)
      throw std::runtime_error("Input array size is not a power of 2.");

    unsigned num_qubits = 0;
    while (array_size > 1) {
      array_size >>= 1;
      num_qubits++;
    }

    if (axes.size() != num_qubits)
      throw std::runtime_error(
          "array size " + std::to_string(array.size()) + " corresponds to "
              + std::to_string(num_qubits) + " qubits but "
              + std::to_string(axes.size()) + " axes are specified.\n");

    //Assign axes backwards to match qsim order
    std::vector<std::string> axes_r(axes.rbegin(), axes.rend());
    init_kstate =
        std::move(KState<Simulator>(num_threads, num_qubits, std::move(axes_r))
        );

    // Manually write each amplitude into the initial state.
    // TODO: figure out how to do this properly with the pointer constructor.
    auto state = init_kstate.active_state();
    for (pybind11::ssize_t ii = 0; ii < array.size(); ii++)
      StateSpace::SetAmpl(state, ii, array.at(ii));

#ifdef DEBUG_SAMPLING
    std::cout << "Initial state:\n";
    init_kstate.print_amplitudes();
    std::cout << "Axes: ";
    for (const auto& ax : init_kstate.qubit_axis)
      std::cout << ax << ", ";
    std::cout << std::endl;
#endif

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
        channel_vec.push_back(std::move(build_koperator(k_op_data)));

      // Assign the KChannel to the register
      auto& reg_vals = regs_channels.first;
      cmap.emplace(reg_vals, channel_vec);
    }

#ifdef DEBUG_SAMPLING
    std::cout << "added KOperation:\n";

    auto k_op = KOperation<fp_type>(cmap, conditional_registers,
                                    is_recorded,
                                    label,
                                    is_virtual);
    k_op.print();
    ops.emplace_back(k_op);
#else
    ops.emplace_back(KOperation<fp_type>(cmap, conditional_registers,
                                         is_recorded,
                                         label,
                                         is_virtual));
#endif

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

    ops.emplace_back(COperation(std::move(channel_map),
                                conditional_registers, std::move(added),
                                is_virtual));

  }

  using RegisterType = uint8_t;
  using OutArrays = std::vector<pybind11::array_t<fp_type>>;
  using AxisOrders = std::vector<std::vector<std::string>>;
  using SamplingOutput = std::tuple<pybind11::array_t<RegisterType>,
                                    OutArrays,
                                    AxisOrders>;
  /** Collect samples from simulation.*/
  SamplingOutput sample_states(size_t num_samples) {
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

    //Initialize an empty 2D numpy array of desired size.
    pybind11::array_t<RegisterType>
        register_mat({num_samples, register_order.size()});
    // A proxy object is needed to access data.
    auto register_mat_proxy = register_mat.mutable_unchecked();

    std::vector<pybind11::array_t<fp_type>> out_arrays;
    out_arrays.reserve(num_samples);

    AxisOrders qubit_axis_orders;
    if (!consistent_axis_order)
      qubit_axis_orders.reserve(num_samples);
    else
      qubit_axis_orders.reserve(1);

    for (size_t ii = 0; ii < num_samples; ii++) {

      k_state.copy_from(init_kstate);
      tmp_state.copy_from(init_kstate);
      final_registers = sample_sequence(ops,
                                        k_state,
                                        tmp_state,
                                        init_registers,
                                        rgen,
                                        virtual_regs);

      //Write final registers
      for (size_t jj = 0; jj < register_order.size(); jj++) {
        *register_mat_proxy.mutable_data(ii, jj) =
            final_registers.at(register_order.at(jj));
      }

      //Allocate output array
      State state = k_state.active_state();
      //We allocate twice as much memory because we are storing a complex vector
      // using two real values.
      unsigned num_qubits = state.num_qubits();
      const unsigned fsv_size = unsigned{1} << (num_qubits + 1);
      auto* fsv = new fp_type[StateSpace::MinSize(num_qubits)];

      // We can only copy the array through a qsim state that shares a pointer
      // to the same allocated memory.
      State dummy_state = StateSpace(num_threads).Create(fsv, num_qubits);
      StateSpace(num_threads).Copy(state, dummy_state);

      //Convert from RRRRRRIIIIII to RIRIRIRIRI representation.
      StateSpace(num_threads).InternalToNormalOrder(dummy_state);

      out_arrays.push_back(as_pyarray({fsv_size}, fsv));

      // record axis order (reverse to account for qsim convention)
      // if axis order is always the same just do this the first time
      if (!consistent_axis_order || ii == 0) {
        const auto
            & axis_order = std::vector<std::string>(k_state.qubit_axis.rbegin(),
                                                    k_state.qubit_axis.rend());
        qubit_axis_orders.push_back(axis_order);
      }
    }

    return std::make_tuple(register_mat, out_arrays, qubit_axis_orders);
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

#endif //QSIM_INTERFACE_INCLUDE_QSIM_KRAUS_SIM_H_
