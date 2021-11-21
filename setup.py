import pathlib

import pkg_resources
from setuptools import find_packages, setup


with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]

setup(
    name="transformer-deploy",
    version=open("./VERSION").read().strip(),
    author="Michaël Benesty",
    description="Simple transformer model optimizer and deployment tool",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ELS-RD/triton_transformers",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    python_requires=">=3.6.0",
    entry_points={
        "console_scripts": [
            "convert_model = transformer_deploy.convert:main",
        ],
    },
)
