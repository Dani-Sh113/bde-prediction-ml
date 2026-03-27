"""
Setup configuration for BDE Prediction ML package.

Installation: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bde-prediction-ml",
    version="1.0.0",
    author="Authors",
    description="Machine learning for C-X Bond Dissociation Energy prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dani-Sh113/bde-prediction-ml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "shap>=0.40.0",
        "tpot>=0.12.0",
        "lightgbm>=3.3.0",
        "joblib>=1.0.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "chemistry": [
            "rdkit>=2021.09.4",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
