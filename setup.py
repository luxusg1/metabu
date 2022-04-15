
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metabu",
    version="0.0.1",
    author="Herilalaina Rakotoarison, Louisot Milijaona, Andry Rasoanaivon, Michele Sebag, and Marc Schoenauer",
    author_email="herilalaina.rakotoarison@inria.fr, milijaonalouisot@gmail.com",
    description="Learning meta-features for AutoML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luxusg1/metabu.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
