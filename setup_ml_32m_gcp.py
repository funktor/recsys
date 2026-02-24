# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension(
        "ml_32m_py",
        sources=["ml_32m_dp_py.pyx", "ml_32m_dp.cpp"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-ltbb"],
        extra_link_args=["-ltbb"],
        include_dirs=[
            "/opt/python/3.10/include/python3.10", 
            "/opt/python/3.10/lib/python3.10/site-packages/torch/include/torch/csrc/api/include",
            "/opt/python/3.10/lib/python3.10/site-packages/torch/include",
            "/usr/include",
            numpy.get_include()
        ],
        library_dirs=["/usr/lib/x86_64-linux-gnu"]
    )
]

setup(
  name = "ML32M",
  version='0.0.16',
  cmdclass = {"build_ext": build_ext},
  ext_modules = cythonize(extensions)
)