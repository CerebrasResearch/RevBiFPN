"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.

File builds RevStructs: reversible structures for use with PyTorch
"""
from setuptools import setup, find_packages

setup(
    name="revstructs",
    version=0.1,
    author="Vitaliy Chiley",
    author_email="vitaliy@cerebras.net, info@cerebras.net",
    url="https://github.com/Cerebras/revbifpn",
    description="RevStructs: Reversible structures for use with PyTorch.",
    packages=find_packages(exclude=("tests")),
)
