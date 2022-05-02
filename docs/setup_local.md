# Installation

!!! tip

    Use Docker if you don't want to install all Nvidia dependencies (for a first try for instance).  
    In the long term, local install is probably a better idea.


!!! danger "6 rules to install locally Nvidia dependencies"

    You may have heard or experienced difficulties in installing Nvidia dependencies, or making them detected by your system.
    If you are on Debian / Ubuntu, it should be ==easy==.  

    **1st rule**: don't follow install guides found on reddit, blogs, etc. they are never up to date

    **2nd rule**:  don't follow install guides from Nvidia dependency manual, they are not always up to date  

    **3rd rule**: only follow install guides from Nvidia ==downlad pages==, they are the only ones with updated instructions  

    **4th rule**: uninstall all your Nvidia dependencies not coming directly from a Nvidia repo (including the Ubuntu driver)  
    and reinstall them from Nvidia repositories  

    **5th rule**: if your OS version is recent and not listed in compatible/tested OS of a dependency, 
    just take the dependency tested latest OS version, it will work otherwise Twitter/forums would be full of complaints.
    
    **6th rule**: choose the network .deb option when possible (meaning add a repo to get updates). Local .deb means manual update.

The list of dependencies you will need to run this library locally:

* [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 11.4.x
* [cuDNN](https://developer.nvidia.com/cudnn-download-survey) 8.2
* [TensorRT](https://developer.nvidia.com/tensorrt) 8.2.1 (GA)

Optional, to run this library from Docker (so you don't have to install all other dependencies):

* [nvidia-docker](https://nvidia.github.io/nvidia-docker/)

You may need to login with a free Nvidia account to download some dependencies.

Then, it's the usual git clone:

```shell
git clone git@github.com:ELS-RD/transformer-deploy.git
cd transformer-deploy
```

* for CPU/GPU support:

```shell
pip3 install ".[GPU]" -f https://download.pytorch.org/whl/cu113/torch_stable.html --extra-index-url https://pypi.ngc.nvidia.com
# if you want to perform GPU quantization (recommended):
pip3 install git+ssh://git@github.com/NVIDIA/TensorRT#egg=pytorch-quantization\&subdirectory=tools/pytorch-quantization/
# if you want to accelerate dense embeddings extraction:
pip install sentence-transformers
```

* for CPU **only** support:

```shell
pip3 install ".[CPU]" -f https://download.pytorch.org/whl/cpu/torch_stable.html
# if you want to accelerate dence embeddings extraction:
pip install sentence-transformers
```

To build your own version of the Docker image:

```shell
make docker_build
```

--8<-- "resources/abbreviations.md"
