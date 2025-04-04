import os
import torch
import glob
import subprocess
from sys import platform

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

    avx_macro = None
    # if on linux, use lscpu command to find if computer has AVX capabilities
    if platform == "linux":
        lscpu_output = subprocess.run(["lscpu"], capture_output=True, text=True)
        cpu_flags = lscpu_output.stdout.split()
        
        if "avx" in cpu_flags and "fma" in cpu_flags and "avx2" in cpu_flags:
            extra_compile_args["cxx"] += ["-mavx", "-mfma", "-mavx2", ]
            avx_macro = [("CPU_IS_AVX_CAPABLE", None)]


    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            define_macros=avx_macro,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.7",
    packages=find_packages(),
    #packages=["qcd_ml_accel_dirac"],
    #package_dir={"qcd_ml_accel_dirac": "qcd_ml_accel_dirac/"},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)

