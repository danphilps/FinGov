#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'FinGov'
DESCRIPTION = 'Utilities for model Governance and Fairness in Finance'
URL = 'https://github.com/danphilps/FinGov.git'
EMAIL = 'danphilps@hotmail.com'
AUTHOR = 'Dan Philps, Madhu Nagarajan, Augusting Backer'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # 'os', 'numpy', 'shap', 'pandas', 'matplotlib.pyplot', 'sklearn'
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(include=['GovernanceUtils.*', 'GovernanceUtils']),
    install_requires=[
        'PyYAML',
        'pandas==0.23.3',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0,,
        'jupyter'
    ]
)
