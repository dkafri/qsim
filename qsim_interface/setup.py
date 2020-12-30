"""Script to compile and install the qsim_kraus_sim pybind interface.

Standard usage:
python setup.py install

To build a "debug" version, do
python setup.py --debug install
this will compile the C++ code with assertions on, which will hurt performance
but may help with catching bugs.

"""

import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

from _version import __version__

description = 'qsim backend interface for kraus_sim simulator'

# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip().split('#')[0] for r in requirements]


class CMakeExtension(Extension):
  def __init__(self, name, sourcedir=''):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


# I do not know of a simpler way of passing the debug command to setup.py.
# Presumably this is not necessary but I don't know how to directly specify
# the debug attribute.
DEBUG = False
if "--debug" in sys.argv:
  DEBUG = True
  sys.argv.remove("--debug")
  __version__ += '-debug'


class CMakeBuild(build_ext):

  def __init__(self, dist):
    super().__init__(dist)
    self.debug = DEBUG

  def run(self):
    try:
      out = subprocess.check_output(['cmake', '--version'])
    except OSError:
      raise RuntimeError(
          "CMake must be installed to build the following extensions: " +
          ", ".join(e.name for e in self.extensions))

    if platform.system() == "Windows":
      cmake_version = LooseVersion(
          re.search(r'version\s*([\d.]+)', out.decode()).group(1))
      if cmake_version < '3.1.0':
        raise RuntimeError("CMake >= 3.1.0 is required on Windows")

    for ext in self.extensions:
      self.build_extension(ext)

  def build_extension(self, ext):
    extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
    # required for auto-detection of auxiliary "native" libs
    if not extdir.endswith(os.path.sep):
      extdir += os.path.sep

    cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                  '-DPYTHON_EXECUTABLE=' + sys.executable]

    cfg = 'Debug' if self.debug else 'Release'
    build_args = ['--config', cfg]

    if platform.system() == "Windows":
      cmake_args += [
          '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
      if sys.maxsize > 2 ** 32:
        cmake_args += ['-A', 'x64']
      build_args += ['--', '/m']
    else:
      cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
      build_args += ['--', '-j2']

    env = os.environ.copy()
    env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
        env.get('CXXFLAGS', ''),
        self.distribution.get_version())
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)
    subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                          cwd=self.build_temp, env=env)
    subprocess.check_call(['cmake', '--build', '.'] + build_args,
                          cwd=self.build_temp)


setup(
    name='qsim_kraus_sim',
    version=__version__,
    author='dkafri@',
    python_requires='>=3.6',
    install_requires=requirements,
    license='',
    description=description,
    cmdclass=dict(build_ext=CMakeBuild),
    ext_modules=[CMakeExtension('qsim_kraus_sim')]
)
