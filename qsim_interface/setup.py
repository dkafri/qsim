from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile

description = 'qsim backend interface for kraus_sim simulator'


# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

ext_modules = [
    Pybind11Extension(
        "qsim_interface",
        ["qsim_interface.cc"],
        cxx_std=14,
        include_dirs=["include", "../lib"]
    ),
]

setup(
    name='qsim_interface',
    version='0.1.0',
    author='dkafri@',
    python_requires='>=3.6',
    license='',
    description=description,
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)
