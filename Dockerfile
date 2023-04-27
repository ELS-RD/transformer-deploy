FROM nvcr.io/nvidia/tritonserver:22.07-py3

# see .dockerignore to check what is transfered

COPY ./ ./

RUN pip3 install --upgrade pip && \
    pip3 install ".[GPU]" --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir && \
    pip3 install sentence-transformers notebook pytorch-quantization ipywidgets
