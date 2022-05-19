"""
Released under BSD 3-Clause License,
Copyright (c) 2022 REDACTED.
All rights reserved.

File builds RevStructs: reversible structures for use with PyTorch
"""
from setuptools import setup, find_packages

setup(
    name="revstructs",
    version=0.1,
    author="REDACTED",
    author_email="REDACTED",
    description="RevStructs: Reversible structures for use with PyTorch.",
    packages=find_packages(exclude=("tests")),
)
