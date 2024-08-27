from setuptools import find_packages, setup

requirements = ["clearml", "nfl-data-py", "pandas", "numpy<2.0.0"]

dev_requirements = ["pre-commit", "black", "flake8", "isort", "pytest"]

setup(
    name="pipeline-cli",
    version="0.7.0",
    author="Jonathan Bailey",
    author_email="jonathan@jelbailey.com",
    description="A CLI tool for managing ClearML Pipelines",
    entry_points={"console_scripts": ["pipeline-cli = src.pipeline_cli:main"]},
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
)
