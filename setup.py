from distutils.core import setup
from Cython.Build import cythonize
import setuptools
import codecs
import os
import re

setup(
    ext_modules = cythonize("bt/core.py")
)


def local_file(filename):
    return codecs.open(
        os.path.join(os.path.dirname(__file__), filename), 'r', 'utf-8'
    )


version = re.search(
    "^__version__ = \((\d+), (\d+), (\d+)\)$",
    local_file('bt/__init__.py').read(),
    re.MULTILINE
).groups()


setuptools.setup(
    name="bt",
    version='.'.join(version),
    author='Philippe Morissette',
    author_email='morissette.philippe@gmail.com',
    description='A flexible backtesting framework for Python',
    keywords='python finance quant backtesting strategies',
    url='https://github.com/pmorissette/bt',
    install_requires=[
        'ffn'
    ],
    packages=['bt'],
    long_description=local_file('README.rst').read(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python'
    ]
)
