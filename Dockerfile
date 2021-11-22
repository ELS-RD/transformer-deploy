FROM nvcr.io/nvidia/tritonserver:21.11-py3

# see .dockerignore to check what is transfered
COPY . ./
RUN pip3 install -U pip && \
    pip3 install nvidia-pyindex && \
    pip3 install .[GPU] -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip3 install . -f https://download.pytorch.org/whl/cu113/torch_stable.html
