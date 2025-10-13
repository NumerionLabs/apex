# setup.py
import os
from setuptools import find_packages, setup

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5 8.6")

with open("requirements.txt") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="apex",
    version="1.0.0",
    author="Atomwise",
    description="APEX source code",
    packages=find_packages(),
    include_package_data=True,
    scripts=[
        "bin/train_factorizer.py",
        "bin/train_surrogate.py",
    ],
    install_requires=["apex_topk"] + install_requires,
    python_requires=">=3.9",
)
