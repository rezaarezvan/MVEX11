#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup, find_packages

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

with open(directory / 'requirements.txt', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='claudeslens',
    version='0.0.1',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=requirements,
    python_requires='>=3.8',
    extras_require={
        'testing': [
            "pytest",
            "tabulate",
        ]
    },
    include_package_data=True
)
