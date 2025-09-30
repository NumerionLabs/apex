# setup.py
import os
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5 8.6")


def get_extensions():
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        # If torch isn't installed, skip extensions entirely
        return [], {}

    # Check if CUDA is available
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    has_cuda = cuda_home is not None and os.path.exists(cuda_home)

    ext_modules = []
    cmdclass = {}

    if has_cuda:
        ext = CUDAExtension(
            name="apex._ops",
            language="c++",
            include_dirs=[
                "include",
                "cccl/cub",
                "cccl/thrust",
                "cccl/libcudacxx/include",
            ],
            sources=[
                "src/library.cc",
                "src/impl-cpu.cc",
                "src/impl-cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )

        ext_modules.append(ext)
        cmdclass = {"build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True,
            use_ninja=False,
        )}

    return ext_modules, cmdclass


ext_modules, cmdclass = get_extensions()

setup(
    name="apex",
    version="0.0.1",
    author="Atomwise, NVIDIA",
    description="APEX source code",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    scripts=[
        "bin/train_factorizer.py",
        "bin/train_surrogate.py",
    ],
    install_requires=[
        "duckdb==1.3.2",
        "numpy==1.23.5",
        "pandas==2.2.3",
        "pyyaml==6.0.1",
        "rdkit==2024.3.5",
        "tensorboard==2.12.1",
        "torch==2.5.1",
    ],
    python_requires=">=3.9",
)
