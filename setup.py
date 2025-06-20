from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="histoSample",
    version="0.1.0",  # Update with your version
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for histological sample analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/histoSample",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.6",  # Adjust based on your requirements
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scikit-image>=0.17.0",
        "opencv-python>=4.2.0",
        "matplotlib>=3.0.0",
        # Add any other dependencies your project needs
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "histosample=histoSample.cli:main",  # If you have a CLI
        ],
    },
    include_package_data=True,  # For non-Python files
)