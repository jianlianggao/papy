#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='papy',
      version='6.0',
      description='Power Analysis tool in Python',
      author='Goncalo Correia, Jianliang Gao',
      author_email='j.gao@imperial.ac.uk',
      url='',
      scripts=["papy/pa.py","papy/plotSurface.py", "papy/runpapy_par.py"],
      install_requires=["numpy","scipy","multiprocessing", "statsmodels", "pandas", "plotly"],
      packages=find_packages(),
      )