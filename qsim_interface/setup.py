from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile
from _version import __version__

description = 'qsim backend interface for kraus_sim simulator'

# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip().split('#')[0] for r in requirements]
print(f'Requirements: {requirements}')

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
    version=__version__,
    author='dkafri@',
    python_requires='>=3.6',
    install_requires=requirements,
    license='',
    description=description,
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)
