from setuptools import find_packages, setup


setup(
    name="ms-stmoe-chla",
    packages=find_packages(exclude=[]),
    version="0.1.0",
    license="MIT",
    description="Multi-scale spatio-temporal mixture-of-experts for marine chlorophyll-a forecasting",
    long_description_content_type="text/markdown",
    keywords=[
        "chlorophyll-a",
        "marine water quality",
        "spatio-temporal forecasting",
        "mixture of experts",
        "pytorch",
    ],
    install_requires=[
        "beartype",
        "CoLT5-attention>=0.10.15",
        "einops>=0.8",
        "einx>=0.3.0",
        "numpy",
        "pandas",
        "torch>=2.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
