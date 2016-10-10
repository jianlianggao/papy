#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='papy',
      version='0.1',
      description='Power Analysis tool in Python',
      author='Gonçalo Correia, Jianliang Gao',
      author_email='j.gao@imperial.ac.uk',
      url='',
      scripts=["papy/pa.py"],
      install_requires=["numpy","scipy","joblib","multiprocessing", "statsmodels", "matplotlib"],
      packages=find_packages(),
      )