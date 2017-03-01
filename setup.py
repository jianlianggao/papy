#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='papy',
      version='3.0',
      description='Power Analysis tool in Python',
      author='Gonc�alo Correia, Jianliang Gao',
      author_email='j.gao@imperial.ac.uk',
      url='',
      py_modules=['papy/pa','papy/plotSurface'],
      install_requires=["numpy","scipy","joblib","multiprocessing", "statsmodels", "matplotlib", "plotly"],
      packages=find_packages(),
      )
