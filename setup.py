#!/usr/bin/env python
import sys
from setuptools import setup
# To use a consistent encoding
from codecs import open
import os
from os import path

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/macaw*")
    sys.exit()

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='macaw',
    packages=['macaw'],
    version='0.2.dev0',
    description='A colorful, long-tailed package for beautiful Majorization-minimization',
    long_description=long_description,
    url='https://github.com/mirca/macaw',
    author='mirca',
    author_email='jvmirca@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
    keywords='statistics probability',
    install_requires=['numpy', 'scipy', 'tensorflow', 'scikit-learn', 'matplotlib']
)
