# Package meta-data.
NAME = 'FinGov'
DESCRIPTION = 'Utilities for model Governance and Fairness in Finance'
URL = 'https://github.com/danphilps/FinGov.git'
EMAIL = 'danphilps@hotmail.com'
AUTHOR = 'Dan Philps, Madhu Nagarajan, Augusting Backer'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

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
