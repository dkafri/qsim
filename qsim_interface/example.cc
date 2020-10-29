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


  py::class_<Matrix<float>>(m, "Matrix", py::buffer_protocol())
      .def_buffer([](Matrix<float>& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(float),                          /* Size of one scalar */
            py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
            2,                                      /* Number of dimensions */
            {m.rows(), m.cols()},                 /* Buffer dimensions */
            {sizeof(float)
                 * m.cols(),             /* Strides (in bytes) for each index */
             sizeof(float)}
        );
      })
      .def(py::init<size_t, size_t>()); // bind constructor

}