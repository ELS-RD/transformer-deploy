FROM nvcr.io/nvidia/tritonserver:22.07-py3

# see .dockerignore to check what is transfered


RUN pip3 install -U pip && \
    pip3 install nvidia-pyindex && \
    pip3 install --pre torch==2.0.0.dev20230128+cu117 --extra-index-url https://download.pytorch.org/whl/nightly/cu117 --no-cache-dir && \
    pip3 install sentence-transformers notebook pytorch-quantization ipywidgets

RUN mkdir /syncback
WORKDIR /transformer_deploy

COPY ./setup.py ./setup.py
COPY ./setup.cfg ./setup.cfg
COPY ./requirements.txt ./requirements.txt
COPY ./src/__init__.py ./src/__init__.py
COPY ./src/transformer_deploy/__init__.py ./src/transformer_deploy/__init__.py


COPY ./ ./
