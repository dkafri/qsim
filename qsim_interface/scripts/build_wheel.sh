#!/usr/bin/env bash

# This script constructs the .whl files required to pip install the
# qsim_kraus_sim package. It should be run within the interface branch.
# In order to compile for a specific python version, make sure to start within
# a virtual environment matching that version. If you get compilation issues,
# try "sudo apt-get install python3.X-dev", where 3.X is your python version.
#
#
# This library is currently housed in a separate github repo for export control
# reasons. If you would like access to it, please message dkafri@.

echo "Creating wheel file for python version"
python3 -V

root=$(git rev-parse --show-toplevel)
files_dir=${root}/qsim_interface/scripts/wheels
mkdir -p ${files_dir}
work_dir="$(mktemp -d '/tmp/qsim-kraus-sim-build-XXXXXXXX')"


git_tag=${1:-$(git rev-parse HEAD)} #specify commit id here
version=0.1.0+${git_tag}


set -e
function clean_up_finally () {
  rm -rf "${work_dir}"
  echo "Cleaning up ${work_dir}..."
}
trap clean_up_finally EXIT

(
  echo "Wheel files will be saved to: ${files_dir}"
  echo "Working directory: ${work_dir}"
  echo "Fetching version ${git_tag}..."
  cd "${work_dir}"
  git clone git@github.com:dkafri/qsim.git qsim
  cd qsim
  git checkout "${git_tag}" --quiet
  git submodule update --init --recursive # to include pybind11
  cd qsim_interface

  echo "building new venv"
  python3 -m venv venv3
  source venv3/bin/activate

  echo "Upgrading pip/wheel"
  pip install --upgrade pip
  pip install --upgrade wheel

#  # Update version in kraus-sim/_version.py.
#  sed -e "s/__version__ = \"[^\"]\+\"/__version__ = \"${version}\"/g" --in-place kraus_sim/_version.py

  echo "installing requirements"
  pip install -r requirements.txt

  # Build wheel
  echo "Building wheel for qsim-kraus-sim"
  python setup.py bdist_wheel --debug
  cd "dist"
  wheel_file=$(ls | grep qsim_kraus_sim)
  cp ${wheel_file} ${files_dir}
  echo "Complete!"



)
