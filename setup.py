#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The setup script."""

from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def get_packages(
    basedirpath: Path,
    file: str,
    search_start: str,
    search_stop: str,
    replace_strings: list,
) -> List[str]:
    """Extract packages from tox.ini as Python list"""
    with open(basedirpath.joinpath(file), "r") as fh:
        packages = []
        for line in fh:
            if search_start in line:
                break

        for line in fh:
            if search_stop in line:
                break
            new_line = line.strip()
            for replace_term in replace_strings:
                if replace_term in new_line:
                    packages.append(new_line.replace(replace_term, ""))

    return packages


basedir = Path(__file__).resolve().parent

# Get long description
with open(basedir.joinpath("DESCRIPTION.rst"), "r") as fh:
    long_description = fh.read()

# Get packages (dependencies)
packages = get_packages(
    basedir, "tox.ini", "lint: pre-commit", "ci: jupyter", ["build: "]
) + ["tox"]

# Get dev packages
dev_packages = packages + ["bumpversion"]

# Get packages required by setup.py
setup_packages = ["check-manifest"]

setup(
    name="chi_crime_predict",
    packages=find_packages(
        exclude=[".pre-commit-config.yaml", ".gitignore"]
    ),
    use_scm_version=True,
    version="0.0.1",
    description="Python DataScience project for crime classification",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Elstan DeSouza",
    author_email="elsdes3@gmail.com",
    url="https://github.com/edesz/chicago-crime-predictor",
    project_urls={
        "Repository": "https://github.com/edesz/chicago-crime-predictor",
        "Issues": "https://github.com/edesz/chicago-crime-predictor/issues",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Topic :: Utilities",
        "Topic :: Software Development",
        "Intended Audience :: Developers",
        "Framework :: tox",
    ],
    # setup dependencies
    setup_requires=setup_packages,
    # production dependencies
    install_requires=packages,
    # optional (development or testing) dependencies
    extras_require={"dev": dev_packages},
    python_requires=">=3.7",
    include_package_data=False,
    zip_safe=False,
    keywords=["python"],
)
