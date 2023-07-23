FROM nvcr.io/nvidia/tritonserver:23.06-py3

# see .dockerignore to check what is transfered

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3-dev \
    python3-distutils \
    python3-venv \
    python3-pip \
    && apt-get clean

ARG UID=10000
ARG GID=10000
RUN addgroup --gid $GID ubuntu && \
    useradd -d /home/ubuntu -ms /bin/bash -g ubuntu -G sudo -u $UID ubuntu
## Switch to ubuntu user by default.
USER ubuntu

WORKDIR /build
RUN pip3 install -U pip --no-cache-dir && \
    pip3 install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117 --no-cache-dir && \
    pip3 install sentence-transformers notebook pytorch-quantization ipywidgets --no-cache-dir

RUN mkdir /syncback
WORKDIR /transformer_deploy

COPY ./setup.py ./setup.py
COPY ./requirements.txt ./requirements.txt
COPY ./requirements_gpu.txt ./requirements_gpu.txt
COPY ./src/__init__.py ./src/__init__.py
COPY ./src/transformer_deploy/__init__.py ./src/transformer_deploy/__init__.py

RUN pip3 install -r requirements.txt && \
    pip3 install nvidia-pyindex --no-cache-dir && \
    pip3 install -r requirements_gpu.txt

COPY ./ ./
