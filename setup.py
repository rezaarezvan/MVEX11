#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(name='bayeslens',
      version='0.0.1',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['bayeslens'],
      classifiers=[
          "Programming Language :: Python :: 3",
      ],
      install_requires=["torch"],
      python_requires='>=3.8',
      extras_require={
          'testing': [
              "pytest",
              "tabulate",
          ]
      },
      include_package_data=True)
