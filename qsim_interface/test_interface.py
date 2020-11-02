import numpy as np
import build.pybind_interface as pbi


def test_uint_matrix_import():
  mat_cpp = pbi.UIntMatrix(2, 3)
  mat_np = np.array(mat_cpp, copy=False)

  assert mat_np.shape == (2, 3)
  assert mat_np.dtype == np.dtype('uint8')

  # Write data
  mat_np.ravel()[:] = np.arange(mat_np.size)

  # confirm both matrices point to same data
  mat_np2 = np.array(mat_cpp, copy=False)
  assert np.array_equal(mat_np, mat_np2)
  assert np.array_equal(np.arange(2 * 3).reshape(2, 3), mat_np)


ComplexType = np.dtype('complex64')


def test_sampler_setters():
  max_qubits = 4
  sampler_cpp = pbi.Sampler(3, max_qubits)
  sampler_cpp.set_random_seed(22)
  sampler_cpp.set_initial_registers({"a": 1})
  sampler_cpp.set_register_order(["c", "d", "e"])

  # state_vec = np.arange((2 ** max_qubits)).astype(ComplexType)
  state_vec = np.zeros((2 ** max_qubits), ComplexType)
  state_vec[3] = 1.0
  state_vec[5] = 2.0
  state_vec[0] = 3.0
  axes = ["a", "b", "c"]
  sampler_cpp.bind_initial_state(state_vec, axes)


def test_add_kop():
  max_qubits = 4
  sampler_cpp = pbi.Sampler(3, max_qubits)

  channels = {
      (0,): [[np.eye(2, dtype=ComplexType).ravel(), [], ["a"], [], [], []]]}
  cond_regs = ("conditional register",)
  is_recorded = False
  is_virtual = False
  label = "ID_a"

  sampler_cpp.add_koperation(channels, cond_regs, is_recorded, label,
                             is_virtual)


def test_add_cop():
  max_qubits = 4
  sampler_cpp = pbi.Sampler(3, max_qubits)

  copy_data = {(0,): (0,),
               (1,): (1,)}
  copy_flip = {(0,): (1,),
               (1,): (0,)}

  channels_map = {(0,): ([[copy_data, ("a",), ("b",)],
                          [copy_flip, ("a",), ("b",)]], [.8, .2]),
                  (1,): ([[copy_data, ("a",), ("b",)],
                          [copy_flip, ("a",), ("b",)]], [.6, .4])
                  }

  sampler_cpp.add_coperation(channels_map, ("c",), {"b"}, False)
