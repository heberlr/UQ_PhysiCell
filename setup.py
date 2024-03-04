from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="uq_physicell",
    version="0.0.2",
    description="project to perform uncertainty quantification of PhysiCell models",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/heberlr/UQ_PhysiCell",
    author="Heber L. Rocha",
    author_email="heberonly@gmail.com",
    keywords="core package",
    license="MIT",
    packages=["uq_physicell"],
    install_requires=[],
    include_package_data=True,
)