//
// Created by dkafri on 10/28/20.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <simmux.h>
#include <pybind_interface.h>
#include <formux.h>

namespace py = pybind11;

template<typename fp_type>
class Matrix {
 public:
  Matrix(size_t rows, size_t cols)
      : m_rows(rows), m_cols(cols), m_data(cols * rows) {};
  fp_type* data() { return m_data.data(); }
  size_t rows() const { return m_rows; }
  size_t cols() const { return m_cols; }
 private:
  size_t m_rows, m_cols;
  std::vector<fp_type> m_data;

};

///** Load a numpy array as an initial state vector to a Sampler.*/


PYBIND11_MODULE(pybind_interface, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring


  using RegisterType = uint8_t;
  py::class_<Matrix<RegisterType>>(m, "UIntMatrix", py::buffer_protocol())
      .def_buffer([](Matrix<RegisterType>& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(RegisterType),                          /* Size of one scalar */
            py::format_descriptor<RegisterType>::format(), /* Python struct-style format descriptor */
            2,                                      /* Number of dimensions */
            {m.rows(), m.cols()},                 /* Buffer dimensions */
            {sizeof(RegisterType)
                 * m.cols(),             /* Strides (in bytes) for each index */
             sizeof(RegisterType)}
        );
      })
      .def(py::init<size_t, size_t>()); // bind constructor


  using Simulator = qsim::Simulator<qsim::For>;
  using Sampler = Sampler<Simulator>;
  py::class_<Sampler>(m, "Sampler")
      .def(py::init<size_t, size_t>()) //bind constructor
      .def("set_random_seed",
           &Sampler::set_random_seed,
           py::arg("seed")) //bind methods
      .def("set_initial_registers",
           &Sampler::set_initial_registers,
           py::arg("registers"))
      .def("set_register_order", &Sampler::set_register_order,
           py::arg("registers"))
      .def("bind_initial_state", &Sampler::bind_initial_state,
           py::arg("array"), py::arg("axes"))
      .def("add_koperation", &Sampler::add_koperation,
           py::arg("channels_map"), py::arg("conditional_registers"),
           py::arg("is_recorded"), py::arg("label"), py::arg("is_virtual"));

}