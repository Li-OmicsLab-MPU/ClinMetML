"""
Setup script for ClinMetML package.

This file is kept for backward compatibility and development installations.
The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="clinmetml",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author="Junrong Li",
    description="Automated Metabolomics Biomarker Discovery and Predictive Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Li-OmicsLab-MPU/ClinMetML",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
        "imbalanced-learn>=0.8.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "rich>=10.0.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "plotly>=5.0.0",
        "shap>=0.40.0",
    ],
    extras_require={
        "deep-learning": [
            "torch>=1.10.0",
            "pytorch-tabular>=1.0.0", 
            "pytorch-lightning>=1.5.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "ruff>=0.0.260",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17",
        ],
    },
    entry_points={
        "console_scripts": [
            "clinmetml=clinmetml.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
