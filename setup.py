import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)

library_name = "qcd_ml_accel_dirac"


def get_extensions():
    extension = CppExtension

    extra_compile_args = {
        "cxx": [
            "-O3",
            "-fopenmp",
        ],
        "nvcc": [
            "-O3",
            "-fopenmp",
        ],
    }

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name)
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    #install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension},
)

