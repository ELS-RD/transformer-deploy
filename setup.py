#  Copyright 2021, Lefebvre Sarrut Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pathlib

import pkg_resources
from setuptools import find_packages, setup


with pathlib.Path("requirements.txt").open() as f:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

with pathlib.Path("requirements_gpu.txt").open() as f:
    extra_gpu = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

with pathlib.Path("requirements_cpu.txt").open() as f:
    extra_cpu = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

setup(
    name="transformer-deploy",
    version=open("./VERSION").read().strip(),
    author="Michaël Benesty",
    author_email="m.benesty@lefebvre-sarrut.eu",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    license_files=("LICENSE",),
    description="Simple transformer model optimizer and deployment tool",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ELS-RD/transformer-deploy",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    extras_require={
        "GPU": extra_gpu,
        "CPU": extra_cpu,
    },
    python_requires=">=3.6.0",
    entry_points={
        "console_scripts": [
            "convert_model = transformer_deploy.convert:entrypoint",
        ],
    },
)
