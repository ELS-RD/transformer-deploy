# Hugging Face Transformer submillisecond inference️ and deployment to production: 🤗 → 🤯

[![Documentation](https://img.shields.io/website?label=documentation&style=for-the-badge&up_message=online&url=https%3A%2F%2Fels-rd.github.io%2Ftransformer-deploy%2F)](https://els-rd.github.io/transformer-deploy/) [![tests](https://img.shields.io/github/workflow/status/ELS-RD/transformer-deploy/tests/main?label=tests&style=for-the-badge)](https://github.com/ELS-RD/transformer-deploy/actions/workflows/python-app.yml) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg?style=for-the-badge)](https://www.python.org/downloads/release/python-360/) [![Twitter Follow](https://img.shields.io/twitter/follow/pommedeterre33?color=orange&style=for-the-badge)](https://twitter.com/pommedeterre33)


### Optimize and deploy in **production** 🤗 Hugging Face Transformer models in a single command line.  

=> Up to 10X faster inference! <=

#### Why this tool?

<!--why-start-->

At [Lefebvre Dalloz](https://www.lefebvre-dalloz.fr/) we run in production *semantic search engines* in the legal domain, 
in non-marketing language it's a re-ranker, and we based ours on `Transformer`.  
In those setup, latency is key to provide good user experience, and relevancy inference is done online for hundreds of snippets per user query.  
We have tested many solutions, and below is what we found:

[`Pytorch`](https://pytorch.org/) + [`FastAPI`](https://fastapi.tiangolo.com/) = 🐢  
Most tutorials on `Transformer` deployment in production are built over Pytorch and FastAPI.
Both are great tools but not very performant in inference (actual measures below).  

[`Microsoft ONNX Runtime`](https://github.com/microsoft/onnxruntime/) + [`Nvidia Triton inference server`](https://github.com/triton-inference-server/server) = ️🏃💨  
Then, if you spend some time, you can build something over ONNX Runtime and Triton inference server.
You will usually get from 2X to 4X faster inference compared to vanilla Pytorch. It's cool!  

[`Nvidia TensorRT`](https://github.com/NVIDIA/TensorRT/)  + [`Nvidia Triton inference server`](https://github.com/triton-inference-server/server) = ⚡️🏃💨💨  
However, if you want the best in class performances on GPU, there is only a single possible combination: Nvidia TensorRT and Triton.
You will usually get 5X faster inference compared to vanilla Pytorch.  
Sometimes it can rise up to **10X faster inference**.  
Buuuuttt... TensorRT can ask some efforts to master, it requires tricks not easy to come up with, we implemented them for you!  

[Detailed tool comparison table](https://els-rd.github.io/transformer-deploy/compare/)

## Features

* Heavily optimize transformer models for inference (CPU and GPU) -> between 5X and 10X speedup
* deploy models on `Nvidia Triton` inference servers (enterprise grade), 6X faster than `FastAPI`
* add quantization support for both CPU and GPU
* simple to use: optimization done in a single command line!
* supported model: any model that can be exported to ONNX (-> most of them)
* supported tasks: classification, feature extraction (aka sentence-transformers dense embeddings)

> Want to understand how it works under the hood?  
> read [🤗 Hugging Face Transformer inference UNDER 1 millisecond latency 📖](https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c?source=friends_link&sk=cd880e05c501c7880f2b9454830b8915)  
> <img src="resources/rabbit.jpg" width="120">

## Want to check by yourself in 3 minutes?

To have a raw idea of what kind of acceleration you will get on your own model, you can try the `docker` only run below.
For GPU run, you need to have installed on your machine Nvidia drivers and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

**3 tasks are covered** below: 

* Classification, 
* feature extraction (text to dense embeddings) 
* text generation (GPT-2 style).  

Moreover, we have added a GPU `quantization` notebook to open directly on `Docker` to play with.

First, clone the repo as some commands below expect to find the `demo` folder:

```shell
git clone git@github.com:ELS-RD/transformer-deploy.git
cd transformer-deploy
# docker image may take a few minutes
docker pull ghcr.io/els-rd/transformer-deploy:0.4.0 
```

### Classification/reranking (encoder model)

Classification is a common task in NLP, and large language models have shown great results.  
This task is also used for search engines to provide Google like relevancy (cf. [arxiv](https://arxiv.org/abs/1901.04085))

#### Optimize existing model

This will optimize models, generate Triton configuration and Triton folder layout in a single command:

```shell
docker run -it --rm --gpus all \
  -v $PWD:/project ghcr.io/els-rd/transformer-deploy:0.4.0 \
  bash -c "cd /project && \
    convert_model -m \"philschmid/MiniLM-L6-H384-uncased-sst2\" \
    --backend tensorrt onnx \
    --seq-len 16 128 128"

# output:  
# ...
# Inference done on NVIDIA GeForce RTX 3090
# latencies:
# [Pytorch (FP32)] mean=5.43ms, sd=0.70ms, min=4.88ms, max=7.81ms, median=5.09ms, 95p=7.01ms, 99p=7.53ms
# [Pytorch (FP16)] mean=6.55ms, sd=1.00ms, min=5.75ms, max=10.38ms, median=6.01ms, 95p=8.57ms, 99p=9.21ms
# [TensorRT (FP16)] mean=0.53ms, sd=0.03ms, min=0.49ms, max=0.61ms, median=0.52ms, 95p=0.57ms, 99p=0.58ms
# [ONNX Runtime (FP32)] mean=1.57ms, sd=0.05ms, min=1.49ms, max=1.90ms, median=1.57ms, 95p=1.63ms, 99p=1.76ms
# [ONNX Runtime (optimized)] mean=0.90ms, sd=0.03ms, min=0.88ms, max=1.23ms, median=0.89ms, 95p=0.95ms, 99p=0.97ms
# Each infence engine output is within 0.3 tolerance compared to Pytorch output
```

It will output mean latency and other statistics.  
Usually `Nvidia TensorRT` is the fastest option and `ONNX Runtime` is usually a strong second option.  
On ONNX Runtime, `optimized` means that kernel fusion and mixed precision are enabled.  
`Pytorch` is never competitive on transformer inference, including mixed precision, whatever the model size.  

#### Run Nvidia Triton inference server

Note that we install `transformers` at run time.  
For production, it's advised to build your own 3-line Docker image with `transformers` pre-installed.

```shell
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:22.01-py3 \
  bash -c "pip install transformers && tritonserver --model-repository=/models"

# output:
# ...
# I0207 09:58:32.738831 1 grpc_server.cc:4195] Started GRPCInferenceService at 0.0.0.0:8001
# I0207 09:58:32.739875 1 http_server.cc:2857] Started HTTPService at 0.0.0.0:8000
# I0207 09:58:32.782066 1 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

#### Query inference 

Query ONNX models (replace `transformer_onnx_inference` by `transformer_tensorrt_inference` to query TensorRT engine):

```shell
curl -X POST  http://localhost:8000/v2/models/transformer_onnx_inference/versions/1/infer \
  --data-binary "@demo/infinity/query_body.bin" \
  --header "Inference-Header-Content-Length: 161"

# output:
# {"model_name":"transformer_onnx_inference","model_version":"1","parameters":{"sequence_id":0,"sequence_start":false,"sequence_end":false},"outputs":[{"name":"output","datatype":"FP32","shape":[1,2],"data":[-3.431640625,3.271484375]}]}
```

Model output is at the end of the Json (`data` field).
[More information about how to query the server from `Python`, and other languages](https://els-rd.github.io/transformer-deploy/run/).

To get very low latency inference in your Python code (no inference server): [click here](https://els-rd.github.io/transformer-deploy/python/)

### Feature extraction / dense embeddings

Feature extraction in NLP is the task to convert text to dense embeddings.  
It has gained some traction as a robust way to improve search engine relevancy (increase recall).  
This project supports models from [sentence-transformers](https://github.com/UKPLab/sentence-transformers).

#### Optimize existing model

```shell
docker run -it --rm --gpus all \
  -v $PWD:/project ghcr.io/els-rd/transformer-deploy:0.4.0 \
  bash -c "cd /project && \
    convert_model -m \"sentence-transformers/msmarco-distilbert-cos-v5\" \
    --backend tensorrt onnx \
    --task embedding \
    --seq-len 16 128 128"

# output:
# ...
# Inference done on NVIDIA GeForce RTX 3090
# latencies:
# [Pytorch (FP32)] mean=5.19ms, sd=0.45ms, min=4.74ms, max=6.64ms, median=5.03ms, 95p=6.14ms, 99p=6.26ms
# [Pytorch (FP16)] mean=5.41ms, sd=0.18ms, min=5.26ms, max=8.15ms, median=5.36ms, 95p=5.62ms, 99p=5.72ms
# [TensorRT (FP16)] mean=0.72ms, sd=0.04ms, min=0.69ms, max=1.33ms, median=0.70ms, 95p=0.78ms, 99p=0.81ms
# [ONNX Runtime (FP32)] mean=1.69ms, sd=0.18ms, min=1.62ms, max=4.07ms, median=1.64ms, 95p=1.86ms, 99p=2.44ms
# [ONNX Runtime (optimized)] mean=1.03ms, sd=0.09ms, min=0.98ms, max=2.30ms, median=1.00ms, 95p=1.15ms, 99p=1.41ms
# Each infence engine output is within 0.3 tolerance compared to Pytorch output
```

#### Run Nvidia Triton inference server

```shell
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:22.01-py3 \
  bash -c "pip install transformers && tritonserver --model-repository=/models"

# output:
# ...
# I0207 11:04:33.761517 1 grpc_server.cc:4195] Started GRPCInferenceService at 0.0.0.0:8001
# I0207 11:04:33.761844 1 http_server.cc:2857] Started HTTPService at 0.0.0.0:8000
# I0207 11:04:33.803373 1 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002

```

#### Query inference 

```shell
curl -X POST  http://localhost:8000/v2/models/transformer_onnx_inference/versions/1/infer \
  --data-binary "@demo/infinity/query_body.bin" \
  --header "Inference-Header-Content-Length: 161"

# output:
# {"model_name":"transformer_onnx_inference","model_version":"1","parameters":{"sequence_id":0,"sequence_start":false,"sequence_end":false},"outputs":[{"name":"output","datatype":"FP32","shape":[1,768],"data":[0.06549072265625,-0.04327392578125,0.1103515625,-0.007320404052734375,...
```

### Generate text (decoder model)

Text generation seems to be the way to go for NLP.  
Unfortunately, they are slow to run, below we will accelerate the most famous of them: GPT-2.

#### Optimize existing model

Like before, command below will prepare Triton inference server stuff.  
One point to have in mind is that Triton run:
- inference engines (`ONNX Runtime` and `TensorRT`)
- `Python` code in charge of the `decoding` part. `Python` code delegate to Triton server the model management.

`Python` code is in `./triton_models/transformer_tensorrt_generate/1/model.py`

```shell
docker run -it --rm --gpus all \
  -v $PWD:/project ghcr.io/els-rd/transformer-deploy:0.4.0 \
  bash -c "cd /project && \
    convert_model -m gpt2 \
    --backend tensorrt onnx \
    --seq-len 6 256 256 \
    --task text-generation"

# output:
# ...
# Inference done on NVIDIA GeForce RTX 3090
# latencies:
# [Pytorch (FP32)] mean=9.43ms, sd=0.59ms, min=8.95ms, max=15.02ms, median=9.33ms, 95p=10.38ms, 99p=12.46ms
# [Pytorch (FP16)] mean=9.92ms, sd=0.55ms, min=9.50ms, max=15.06ms, median=9.74ms, 95p=10.96ms, 99p=12.26ms
# [TensorRT (FP16)] mean=2.19ms, sd=0.18ms, min=2.06ms, max=3.04ms, median=2.10ms, 95p=2.64ms, 99p=2.79ms
# [ONNX Runtime (FP32)] mean=4.99ms, sd=0.38ms, min=4.68ms, max=9.09ms, median=4.78ms, 95p=5.72ms, 99p=5.95ms
# [ONNX Runtime (optimized)] mean=3.93ms, sd=0.40ms, min=3.62ms, max=6.53ms, median=3.81ms, 95p=4.49ms, 99p=5.79ms
# Each infence engine output is within 0.3 tolerance compared to Pytorch output
```

#### Run Nvidia Triton inference server

To run decoding algorithm server side, we need to install `Pytorch` on `Triton` docker image.

```shell
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 8g \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:22.01-py3 \
  bash -c "pip install transformers torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
  tritonserver --model-repository=/models"

# output:
# ...
# I0207 10:29:19.091191 1 grpc_server.cc:4195] Started GRPCInferenceService at 0.0.0.0:8001
# I0207 10:29:19.091417 1 http_server.cc:2857] Started HTTPService at 0.0.0.0:8000
# I0207 10:29:19.132902 1 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

#### Query inference 

Replace `transformer_onnx_generate` by `transformer_tensorrt_generate` to query `TensorRT` engine.

```shell
curl -X POST  http://localhost:8000/v2/models/transformer_onnx_generate/versions/1/infer \
  --data-binary "@demo/infinity/query_body.bin" \
  --header "Inference-Header-Content-Length: 161"

# output:
# {"model_name":"transformer_onnx_generate","model_version":"1","outputs":[{"name":"output","datatype":"BYTES","shape":[],"data":["This live event is great. I will sign-up for Infinity.\n\nI'm going to be doing a live stream of the event.\n\nI"]}]}
```

Ok, the output is not very interesting (💩 in -> 💩 out) but you get the idea.  
Source code of the generative model is in `./triton_models/transformer_tensorrt_generate/1/model.py`.  
You may want to tweak it regarding your needs (default is set for greedy search and output 64 tokens).

#### Python code

You may be interested in running optimized text generation on Python directly, without using any inference server:  

```shell
docker run -p 8888:8888 -v $PWD/demo/generative-model:/project ghcr.io/els-rd/transformer-deploy:0.4.0 \
  bash -c "cd /project && jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root"
```

### Model quantization on GPU

Quantization is a generic method to get X2 speedup on top of other inference optimization.  
GPU quantization on transformers is almost never used because it requires to modify model source code.  

We have implemented in this library a mechanism which updates Hugging Face transformers library to support quantization.  
It makes it easy to use.

To play with it, open this notebook:

```shell
docker run -p 8888:8888 -v $PWD/demo/quantization:/project ghcr.io/els-rd/transformer-deploy:0.4.0 \
  bash -c "cd /project && jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root"
```

<!--why-end-->

## See our [documentation](https://els-rd.github.io/transformer-deploy/) for detailed instructions on how to use the package, including setup, GPU quantization support and Nvidia Triton inference server deployment.
