//
// Created by dkafri on 10/28/20.
//

#include <pybind11/pybind11.h>

#include <utility>

namespace py = pybind11;

int add(int i, int j) {
  return i + j;
}

struct Pet {
  explicit Pet(std::string name) : name(std::move(name)), age(0) {}
  void setName(const std::string& name_) { name = name_; }
  const std::string& getName() const { return name; }

  std::string name;
  int age;
};

PYBIND11_MODULE(pybind_example, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("add", &add, "A function which adds two numbers",
        py::arg("i") = 1, py::arg("j") = 2);

  py::class_<Pet>(m, "Pet")
      .def(py::init<std::string>()) //bind constructor
      .def("setName", &Pet::setName) //bind method
      .def("getName", &Pet::getName)
      .def("__repr__", // bind lambda function
           [](const Pet& a) {
             return "<example.Pet named '" + a.name + "'>";
           })
      .def_readwrite("age", &Pet::age); // give attribute access

}