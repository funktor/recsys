# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
)

setup(
    name="extension_cpp",
    version="0.0.2",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "extension_cpp",
            ["pytorch_c_ext.cpp", "mysoftmax.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-ltbb"]
            },
            extra_link_args=["-ltbb"],
            include_dirs=[
                "/opt/python/3.10/include/python3.10",
                "/usr/include"
            ],
            library_dirs=["/usr/lib/x86_64-linux-gnu/"]
        )
    ],
    install_requires=["torch"],
    description="Example of PyTorch C++ and CUDA extensions",
    cmdclass={"build_ext": BuildExtension}
)
