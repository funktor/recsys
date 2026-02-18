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
            "/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13", 
            "/Users/amondal/recsys/.venv/lib/python3.13/site-packages/torch/include/torch/csrc/api/include",
            "/Users/amondal/recsys/.venv/lib/python3.13/site-packages/torch/include",
            "/opt/homebrew/opt/tbb/include",
            numpy.get_include()
        ],
        library_dirs=["/opt/homebrew/opt/tbb/lib"]
    )
]

setup(
  name = "ML32M",
  version='0.0.16',
  cmdclass = {"build_ext": build_ext},
  ext_modules = cythonize(extensions)
)