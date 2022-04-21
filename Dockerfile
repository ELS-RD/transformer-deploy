FROM nvcr.io/nvidia/tritonserver:22.02-py3

RUN apt-get update
RUN apt-get install -y pkg-config  
RUN apt-get install -y libcairo2-dev libjpeg-dev libgif-dev

RUN mkdir /reqs
ADD requirements* /reqs/
RUN pip install -r /reqs/requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir
RUN pip install -r /reqs/requirements_gpu.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir

RUN pip3 install -U pip && \
    pip3 install nvidia-pyindex && \
    pip3 install sentence-transformers notebook pytorch-quantization

# see .dockerignore to check what is transfered
COPY . ./

RUN pip3 install ".[GPU]" -f https://download.pytorch.org/whl/cu113/torch_stable.html --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir

WORKDIR /project