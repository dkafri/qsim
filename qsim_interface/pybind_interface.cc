//
// Created by dkafri on 10/28/20.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <simmux.h>
#include <pybind_interface.h>
#include <formux.h>

namespace py = pybind11;


///** Load a numpy array as an initial state vector to a Sampler.*/


PYBIND11_MODULE(pybind_interface, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring





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
           py::arg("is_recorded"), py::arg("label"), py::arg("is_virtual"))
      .def("add_coperation",
           &Sampler::add_coperation,
           py::arg("channels_map"),
           py::arg("conditional_registers"),
           py::arg("added_registers"),
           py::arg("is_virtual"))
      .def("sample_states", &Sampler::sample_states,
           py::arg("num_samples"));


  using RegisterType = Sampler::RegisterType;
  py::class_<MatrixBuffer<RegisterType>>(m, "UIntMatrix", py::buffer_protocol())
      .def_buffer([](MatrixBuffer<RegisterType>& m) -> py::buffer_info {
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

}