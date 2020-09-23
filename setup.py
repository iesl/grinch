# !/usr/bin/env python

from distutils.core import setup

setup(name='grinch',
      version='0.01',
      packages=['grinch'],
      install_requires=[
          "wandb",
          "absl-py",
          "mysql-connector-python",
      ],
      package_dir={'grinch': 'src/python/grinch'}
      )