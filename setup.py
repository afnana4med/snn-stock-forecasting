from setuptools import setup, find_packages

setup(
    name="snn_stock",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "pyyaml",
        "matplotlib",
    ]
)