from setuptools import setup, find_packages

setup(
    name="weightboost",
    version="0.1.0",
    author="COMP 7404",
    author_email="user@example.com",
    description="Implementation of the WeightBoost algorithm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/weightboost",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "pandas>=1.1.0",
    ],
)
