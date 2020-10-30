//
// Created by dkafri on 10/28/20.
//

#include <pybind11/pybind11.h>

#include <utility>
#include <simmux.h>
#include <sampling.h>
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
      .def("set_random_seed", &Sampler::set_random_seed);
}