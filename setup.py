from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Fourier_Tool",
    version="0.1.0",
    author="Daksh Saini",
    author_email="daksh.saini@example.com",  # Replace with your actual email if desired
    description="A domain-agnostic framework for signal filtering and denoising using Fourier analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arachknight-daksh/Fourier_Tool", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    # This tells setuptools that your code is under 'src'
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.25.0",
        "scipy>=1.12.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0"
    ],
)