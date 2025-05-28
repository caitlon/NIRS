"""
Setup file for installing the project as a package.
"""

from setuptools import setup, find_packages

setup(
    name="nirs-tomato",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "joblib>=1.2.0",
        "mlflow>=2.11.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'isort>=5.10.0',
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'openpyxl>=3.1.0',
            'statsmodels>=0.14.0',
            'tqdm>=4.65.0',
            'boto3>=1.34.0',
        ],
    },
    python_requires='>=3.9',
    description="Library for processing and analyzing NIR spectroscopy data of tomatoes for Brix prediction",
    author="NIRS Tomato Team",
) 