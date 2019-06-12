from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  #'tensorflow-gpu',
  #'tensorflow-gpu==2.0.0-alpha0',
  'dltk',
  'nibabel',
  'numpy>=1.14.2',
  'pandas>=0.23.4',
  'six'
]

setup(
    name='preprocessing',
    version='0.1',
    author='Kim Milam',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages()
)