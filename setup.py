import pathlib

import pkg_resources
from setuptools import find_packages, setup


with pathlib.Path("requirements.txt").open() as f:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

with pathlib.Path("requirements_gpu.txt").open() as f:
    extra_gpu = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]


setup(
    name="transformer-deploy",
    version=open("./VERSION").read().strip(),
    author="Michaël Benesty",
    author_email="m.benesty@lefebvre-sarrut.eu",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    description="Simple transformer model optimizer and deployment tool",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ELS-RD/triton_transformers",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    extras_require={
        "GPU": extra_gpu,
    },
    python_requires=">=3.6.0",
    entry_points={
        "console_scripts": [
            "convert_model = transformer_deploy.convert:main",
        ],
    },
)
