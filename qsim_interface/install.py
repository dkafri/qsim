"""Python script that invokes setup.py with optional arguments.

Note: If you are installing from source, you must also pull the pybind
submodule before running setup.py. Within the qsim_interface directory, do:
git submodule update --init --recursive

Standard install (from source):
python install.py

Requirements: cmake version 3.13+, git.

For options, do
python install.py -h

"""
import argparse
import os
import re
from typing import Optional
from warnings import warn

parser = argparse.ArgumentParser()
parser.add_argument('--python-version', type=str, default=None,
                    help="Which python version to compile the library with. If "
                         "not specified CMake will likely use the most recent"
                         " version.")
parser.add_argument('--debug', help='Compile in debug mode. The compiled code '
                                    'evaluates assertions, which can be useful '
                                    'for debugging, but comes with a minor '
                                    'performance cost.',
                    action='store_true')

CONFIG_FILE = 'setup.cfg'


def main(debug: bool, python_version: Optional[str]):
    """Install the qsim_kraus_sim package.

    To see documentation, do: `python install.py -h`

    """

    # confirm python_version is correct format
    if python_version is not None:
        expected_format = '3\.([6-9]|1[0-9])(\.[0-9]+)*'
        match = re.search(expected_format, python_version)
        if not match or match.end() - match.start() != len(python_version):
            raise ValueError(f'Require a python version at least 3.6, e.g. '
                             f'"3.7.2"')

    # We create a temporary setup.cfg with the desired options, then run
    # setup.py install.
    if os.path.exists(CONFIG_FILE):
        warn(f'Config file already exists at {CONFIG_FILE}. This command will'
             f' now overwrite it.')
        os.remove(CONFIG_FILE)

    with open(CONFIG_FILE, 'w') as config_file:
        config_file.write('[build_ext]\n')
        config_file.write(f'debug={int(debug)}\n')
        if python_version is not None:
            config_file.write(f'python_version={python_version}')

    os.system('python3 setup.py install')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.debug, args.python_version)
