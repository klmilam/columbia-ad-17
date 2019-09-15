from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'dltk',
    'nibabel',
    'six',
    'tensorflow-transform',
    'apache-beam[gcp]'
]

setup(
    name='preprocessing',
    version='0.1',
    author='Kim Milam',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages()
)