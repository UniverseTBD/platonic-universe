#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="platonic-universe",
    version="0.1.0",
    author="Platonic Universe Team",
    author_email="team@platonic-universe.org",
    description="A modular Python package for multimodal astronomical data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/platonic-universe/platonic-universe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.21.0",
        "timm>=0.9.0",
        "huggingface-hub>=0.16.0",
        "datasets>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pillow>=8.0.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.5.0",
        "astropy>=5.0.0",
    ],
    extras_require={
        "spectral": [
            "specutils>=1.8.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "platonic-universe=platonic_universe.cli:main",
        ],
    },
)