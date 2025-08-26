# Standard
import os

# Third party
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5 8.6"

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

build_ext = BuildExtension.with_options(
    no_python_abi_suffix=True,
    use_ninja=False,
)

setup(
    name="apex",
    version="0.0.1",
    author="Atomwise, NVIDIA",
    description="APEX source code",

    packages=find_packages(),
    include_package_data=True,
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    scripts=[
        "bin/train_apex.py",
        "bin/train_surrogate.py",
    ],

    install_requires=[
        "numpy==1.23.5",
        "pandas==2.2.3",
        "pyyaml==6.0.1",
        "rdkit==2024.3.5",
        "tensorboard==2.12.1",
        "torch==2.5.1",
    ],
    python_requires=">=3.9",
)
