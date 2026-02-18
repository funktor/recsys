# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)

setup(
    name="extension_cpp",
    version="0.1.2",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            "extension_cpp",
            ["pytorch_c_ext.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-ltbb", "-Wall"]
            },
            extra_link_args=["-ltbb"],
            include_dirs=[
                "/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13", 
                "/Users/amondal/recsys/.venv/lib/python3.13/site-packages/torch/include/torch/csrc/api/include",
                "/Users/amondal/recsys/.venv/lib/python3.13/site-packages/torch/include",
                "/opt/homebrew/opt/tbb/include"
            ],
            library_dirs=["/opt/homebrew/opt/tbb/lib"]
        )
    ],
    install_requires=["torch"],
    description="Example of PyTorch C++ and CUDA extensions",
    cmdclass={"build_ext": BuildExtension}
)