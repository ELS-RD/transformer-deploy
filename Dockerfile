FROM nvcr.io/nvidia/tritonserver:21.10-py3

COPY requirements.txt requirements.txt
RUN pip3 install -U pip && \
    pip3 install nvidia-pyindex && \
    pip3 install -r requirements.txt
