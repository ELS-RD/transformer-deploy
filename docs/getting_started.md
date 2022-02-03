# Getting started

Below we provide shortcuts to test the project without having to install it.  


## Classification/reranking (encoder model)

Classification is a common task in NLP, and large language models have shown great results.  
This task is also used in search to provide Google like relevancy (cf. [arxiv](https://arxiv.org/abs/1901.04085))

### Optimize model

```shell
docker run -it --rm --gpus all \
  -v $PWD:/project ghcr.io/els-rd/transformer-deploy:0.3.0 \
  bash -c "cd /project && \
    convert_model -m \"philschmid/MiniLM-L6-H384-uncased-sst2\" \
    --backend tensorrt onnx \
    --seq-len 16 128 128"
```

### Run Nvidia Triton inference server

```shell
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:21.12-py3 \
  bash -c "pip install transformers && tritonserver --model-repository=/models"
```

### Query inference 

```shell
curl -X POST  http://localhost:8000/v2/models/transformer_onnx_inference/versions/1/infer \
  --data-binary "@demo/infinity/query_body.bin" \
  --header "Inference-Header-Content-Length: 161"
```

## Feature extraction (sentence-transformers model)

### Optimize model

```shell

```

### Run Nvidia Triton inference server

```shell

```

### Query inference 

```shell
curl -X POST  http://localhost:8000/v2/models/transformer_onnx_inference/versions/1/infer \
  --data-binary "@demo/infinity/query_body.bin" \
  --header "Inference-Header-Content-Length: 161"
```

## Generate text (decoder model)


# pip install transformers torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# tritonserver --model-repository=/models


### Optimize model

```shell

```

### Run Nvidia Triton inference server

```shell
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 8g \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:21.12-py3 -bash -c \
  "pip install transformers torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
  tritonserver --model-repository=/models"
```

### Query inference 

```shell
curl -X POST  http://localhost:8000/v2/models/transformer_onnx_inference/versions/1/infer \
  --data-binary "@demo/infinity/query_body.bin" \
  --header "Inference-Header-Content-Length: 161"
```