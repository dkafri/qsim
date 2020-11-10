//
// Created by dkafri on 10/28/20.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <simmux.h>
#include <formux.h>
#include <qsim_kraus_sim.h>

namespace py = pybind11;



PYBIND11_MODULE(qsim_kraus_sim, m) {
  m.doc() =
      "wrapper for qsim trajectories functionality"; // optional module docstring


  using Simulator = qsim::Simulator<qsim::For>;
  using Sampler = Sampler<Simulator>;
  py::class_<Sampler>(m, "Sampler")
      .def(py::init<size_t, bool>(),
           py::arg("num_threads"),
           py::arg("consistent_axis_order")) //bind constructor
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

}