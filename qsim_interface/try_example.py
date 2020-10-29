"""Try importing C++ interface module."""
import numpy as np
import build.pybind_example as example

print(example.add(2, 3))

mat_cpp = example.Matrix(2, 3)
mat_np = np.array(mat_cpp, copy=False)
mat_np2 = np.array(mat_cpp, copy=False)

mat_np.ravel()[:] = np.arange(mat_np.size)
assert np.all(mat_np == mat_np2)
