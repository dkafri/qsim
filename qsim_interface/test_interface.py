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


def test_sampler_setters():
  sampler_cpp = pbi.Sampler(3, 4)
  sampler_cpp.set_random_seed(22)
  sampler_cpp.set_initial_registers({"a": 1})
  sampler_cpp.set_register_order(["c", "d", "e"])
