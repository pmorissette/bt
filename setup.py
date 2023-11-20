import codecs
import os

from setuptools import setup
from setuptools.extension import Extension


def local_file(filename):
    return codecs.open(os.path.join(os.path.dirname(__file__), filename), "r", "utf-8")


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
    version="1.0.0",
    author="Philippe Morissette",
    author_email="morissette.philippe@gmail.com",
    description="A flexible backtesting framework for Python",
    keywords="python finance quant backtesting strategies algotrading algorithmic trading",
    url="https://github.com/pmorissette/bt",
    license="MIT",
    install_requires=["ffn>=1.0.0", "pyprind>=2.11"],
    extras_require={
        "dev": [
            "cython>=0.25",
            "ffn>=1.0.0",
            "matplotlib>=2",
            "numpy>=1",
            "pandas>=0.19",
            "pyprind>=2.11",
            "pytest",
            "pytest-cov",
            "ruff",
        ],
    },
    packages=["bt"],
    long_description=local_file("README.md").read().replace("\r\n", "\n"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
    ],
    ext_modules=ext_modules,
    python_requires=">=3.8",
)
