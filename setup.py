from __future__ import annotations

import os
import sys

from setuptools import setup


def cuda_extensions():
    if os.environ.get("TEXELATOR_BUILD_CUDA", "").upper() not in {"1", "ON", "YES", "TRUE"}:
        return [], {}
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as error:
        raise RuntimeError(
            "building the native Texelator wheel requires CUDA-enabled PyTorch in the build environment"
        ) from error
    extension = CUDAExtension(
        "texelator._cuda",
        sources=[
            "texelator/cuda/extension.cpp",
            "texelator/cuda/texelator_cuda.cu",
        ],
        extra_compile_args={
            "cxx": (["/O2"] if sys.platform == "win32" else ["-O3"]),
            "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
        },
    )
    return [extension], {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}


extensions, commands = cuda_extensions()
setup(ext_modules=extensions, cmdclass=commands)
