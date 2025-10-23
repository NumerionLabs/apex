# Third party
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="apex",
    version="1.0.0",
    author="Numerion Labs",
    description="APEX source code",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["apex_topk"] + install_requires,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "prepare_library=apex.cli.prepare_library:main",
            "run_search=apex.cli.run_search:main",
            "train_factorizer=apex.cli.train_factorizer:main",
            "train_surrogate=apex.cli.train_surrogate:main",
        ],
    },
)
