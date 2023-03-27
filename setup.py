import sys
from setuptools import setup

setup(
    name="he_preprocessing",
    version="0.1.0",
    packages=[
        "he_preprocessing",
        "he_preprocessing.utils",
        "he_preprocessing.filter",
        "he_preprocessing.normalization",
        "he_preprocessing.quality_control",
    ],
    url="https://github.com/tsikup/he_preprocessing",
    license="MIT",
    author="Nikos Tsiknakis",
    author_email="tsiknakisn@gmail.com",
    description=" A collection of preprocessing tools for H&E slide analysis, developed and used for my PhD project.",
    install_requires=[
        "PyYAML",
        "dotmap",
        "natsort",
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "tqdm",
        "opencv-python",
        "scikit-image",
        "Pillow",
        "shapely",
    ]
    + (["spams"] if sys.platform == "darwin" else ["spams-bin"]),
)
