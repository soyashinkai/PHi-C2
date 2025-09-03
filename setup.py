from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="phic",
    version="2.1.3",
    author="Soya SHINKAI and Soya HAGIWARA",
    author_email="soya.shinkai@riken.jp",
    license="GPL-3.0",
    description="Polymer dynamics simulation from Hi-C data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/soyashinkai/PHi-C2",
    keywords=["biophysics", "bioinformatics", "genomics", "Hi-C",
              "polymer modeling", "polymer dynamics", "rheology", ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "matplotlib", "scipy", "click", "pandas", "hic-straw", "cooler", "h5py",],
    entry_points={
        "console_scripts": [
            "phic = src.phic2:cli",
        ]
    }
)
