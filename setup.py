#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='papy',
      version='5.0',
      description='Power Analysis tool in Python',
      author='Gonc¸alo Correia, Jianliang Gao',
      author_email='j.gao@imperial.ac.uk',
      url='',
      scripts=["papy/pa.py","papy/plotSurface.py"],
      install_requires=["numpy","scipy","multiprocessing", "statsmodels", "plotly"],
      packages=find_packages(),
      )
