FROM nvcr.io/nvidia/tritonserver:22.07-py3

WORKDIR /build

COPY setup.py .
COPY requirements.txt .
COPY requirements_gpu.txt .
COPY src/ src/

RUN pip3 install --no-cache-dir -r requirements_gpu.txt && \
    pip3 install --no-cache-dir -r requirements.txt && \
    python3 setup.py install \

# see .dockerignore to check what is transfered
RUN mkdir /syncback
WORKDIR /transformer_deploy

# add python bin to the PATH env variable
ENV PATH=".local/bin/:${PATH}"

RUN pip3 install -U pip && \
    pip3 install nvidia-pyindex && \
    pip3 install --pre torch==2.0.0.dev20230128+cu117 --extra-index-url https://download.pytorch.org/whl/nightly/cu117 --no-cache-dir && \
    pip3 install sentence-transformers notebook pytorch-quantization ipywidgets

COPY ./ ./
