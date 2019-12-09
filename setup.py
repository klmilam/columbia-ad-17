from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow-transform==0.14.0', 'gcsfs==0.4.0', 'pandas']

setup(
    name='MRI_TPU_Trainer',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='CNN on TPU using Cloud AI Platform.')