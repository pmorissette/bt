from setuptools import setup
from klink import __version__

setup(
    name='klink',
    version=__version__,
    url='https://github.com/pmorissette/klink',
    description='Klink is a simple and clean theme for creating Sphinx docs, inspired by jrnl',
    license='MIT',
    author='Philippe Morissette',
    author_email='morissette.philippe@gmail.com',
    packages=['klink']
)
