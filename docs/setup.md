# Installation

??? tip "Required dependencies"

    To install this package locally, you need:
    
    **TensorRT**
    
    * [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) v8.2.1 (GA)
    
    **System Packages**
    
    * [CUDA](https://developer.nvidia.com/cuda-toolkit)
      * Recommended versions:
      * cuda-11.4.x + cuDNN-8.2
      * cuda-10.2 + cuDNN-8.2
    * [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
    * [cmake](https://github.com/Kitware/CMake/releases) >= v3.13
    * [python](<https://www.python.org/downloads/>) >= v3.6.9
    * [pip](https://pypi.org/project/pip/#history) >= v19.0

    To be able to leverage your CUDA installation by the Pycuda dependency, 
    don't forget to add to your $PATH env variable the path to CUDA. Otherwise, Pycuda will fail to compile.

```shell
git clone git@github.com:ELS-RD/transformer-deploy.git
cd transformer-deploy
```

* for GPU support:

```shell
pip3 install ".[GPU]" -f https://download.pytorch.org/whl/cu113/torch_stable.html --extra-index-url https://pypi.ngc.nvidia.com
# if you want to perform GPU quantization (recommended)
pip3 install git+ssh://git@github.com/NVIDIA/TensorRT#egg=pytorch-quantization\&subdirectory=tools/pytorch-quantization/
```

* for CPU support:

```shell
pip3 install ".[CPU]" -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

To build your own version of the Docker image:

```shell
make build_docker
```