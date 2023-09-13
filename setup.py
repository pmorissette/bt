import codecs
import os
import re

from setuptools import setup
from setuptools.extension import Extension


def local_file(filename):
    return codecs.open(os.path.join(os.path.dirname(__file__), filename), "r", "utf-8")


version = re.search(
    "^__version__ = \\((\\d+), (\\d+), (\\d+)\\)",
    local_file(os.path.join("bt", "__init__.py")).read(),
    re.MULTILINE,
).groups()

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

ext_modules = []

if use_cython:
    ext_modules = cythonize("bt/core.py")
else:
    ext_modules = [Extension("bt.core", ["bt/core.c"])]

setup(
    name="bt",
    version=".".join(version),
    author="Philippe Morissette",
    author_email="morissette.philippe@gmail.com",
    description="A flexible backtesting framework for Python",
    keywords="python finance quant backtesting strategies",
    url="https://github.com/pmorissette/bt",
    license="MIT",
    install_requires=[
        "ffn>=0.3.7",
        "pyprind>=2.11",
        "tqdm==4.65.0"  # <-- Added this line
    ],
    extras_require={
        "dev": [
            "black>=20.8b1",
            "codecov",
            "cython>=0.25",
            "ffn>=0.3.5",
            "flake8",
            "flake8-black",
            "matplotlib>=2",
            "numpy>=1",
            "pandas>=0.19",
            "pyprind>=2.11",
            "pytest",
            "pytest-cov",
        ],
    },
    packages=["bt"],
    long_description=local_file("README.rst").read().replace("\r\n", "\n"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python",
    ],
    ext_modules=ext_modules,
    python_requires=">=3.7",
)
