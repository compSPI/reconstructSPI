"""Create instructions to build the reconstructSPI package."""

import setuptools

requirements = []

setuptools.setup(
    name="reconstructSPI",
    maintainer="Frederic Poitevin",
    version="0.0.1",
    maintainer_email="frederic.poitevin@stanford.edu",
    description="Reconstruction methods and tools for SPI",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/compSPI/reconstructSPI.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
