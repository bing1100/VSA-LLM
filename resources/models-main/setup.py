from distutils.core import setup
from setuptools import find_packages

setup(
    name="medair-models",
    version="1.0",
    packages=find_packages("."),
    author="Trevor Yu",
    author_email="tcwyu@uwaterloo.ca",
    url="https://git.uwaterloo.ca/medical-ai-lab/models/",
    install_requires=["transformers"]
)