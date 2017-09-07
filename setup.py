#!/usr/bin/env python
from setuptools import setup, find_packages
from datetime import datetime

# Use a date-stamp format for versioning
now = datetime.now()
VERSION = now.strftime("%Y-%m-%d")

NAME = 'stat_pm25'
LICENSE = 'BSD 3-Clause'
AUTHOR = 'Daniel Rothenberg'
AUTHOR_EMAIL = 'darothen@mit.edu'

setup(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR, author_email=AUTHOR_EMAIL,
    packages=find_packages()r,
    # package_dir={'': 'src'},
    scripts=[],
)
