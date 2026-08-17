
from setuptools import setup, find_packages

setup(
    name="cytof_helper",
    version="0.1.0",
    description="Helper functions for CyTOF data analysis, normalization, and visualization",
    author="Ronguy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "umap-learn",
        "plotly",
        "ipywidgets",
        "torch",
        "tqdm",
        "lmfit",  # Required for optimization-based normalization
        "xgboost", # Required for interactive classifier
    ],
)
