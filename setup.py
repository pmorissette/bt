from Cython.Build import cythonize
from distutils.core import setup
import codecs
import os
import re


def local_file(filename):
    return codecs.open(
        os.path.join(os.path.dirname(__file__), filename), 'r', 'utf-8'
    )

version = re.search(
    "^__version__ = \((\d+), (\d+), (\d+)\)$",
    local_file('bt/__init__.py').read(),
    re.MULTILINE
).groups()

setup(
    name="bt",
    version='.'.join(version),
    author='Philippe Morissette',
    author_email='morissette.philippe@gmail.com',
    description='A flexible backtesting framework for Python',
    keywords='python finance quant backtesting strategies',
    url='https://github.com/pmorissette/bt',
    requires=[
        'ffn'
    ],
    packages=['bt'],
    long_description=local_file('README.rst').read(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python'
    ],
    ext_modules=cythonize('bt/core.py')
)
