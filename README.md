# Hugging Face Transformer submillisecond inference️ and deployment to production: 🤗 → 🤯

[![tests](https://github.com/ELS-RD/transformer-deploy/actions/workflows/python-app.yml/badge.svg)](https://github.com/ELS-RD/transformer-deploy/actions/workflows/python-app.yml) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENCE) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

### Optimize and deploy in **production** Hugging Face Transformer models in a single command line.  

=> Up to 10X faster inference! <=

#### Table of Contents

* [🤔 why this tool?](#why-this-tool)
* [🏗️ Installation](#installation) 
* [🤓 run (1 command)](#run-in-a-single-command)
* [🐍 TensorRT usage in Python script](#tensorrt-usage-in-python-script)
* [⏱ benchmarks](#benchmarks)
* [🤗 end to end reproduction of Infinity Hugging Face demo](./demo/README.md) (to replay [Medium article](https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c?source=friends_link&sk=cd880e05c501c7880f2b9454830b8915))

#### Why this tool?

[`Pytorch`](https://pytorch.org/) + [`FastAPI`](https://fastapi.tiangolo.com/) = 🐢  
Most tutorials on transformer deployment in production are built over Pytorch and FastAPI.
Both are great tools but not very performant in inference.  

[`Microsoft ONNX Runtime`](https://github.com/microsoft/onnxruntime/) + [`Nvidia Triton inference server`](https://github.com/triton-inference-server/server) = ️🏃💨  
Then, if you spend some time, you can build something over ONNX Runtime and Triton inference server.
You will usually get from 2X to 4X faster inference compared to vanilla Pytorch. It's cool!  

[`Nvidia TensorRT`](https://github.com/NVIDIA/TensorRT/)  + [`Nvidia Triton inference server`](https://github.com/triton-inference-server/server) = ⚡️🏃💨💨  
However, if you want the best in class performances on GPU, there is only a single possible combination: Nvidia TensorRT and Triton.
You will usually get 5X faster inference compared to vanilla Pytorch. 
Sometimes it can raises up to **10X faster inference**.
Buuuuttt... TensorRT is not easy to use, even less with Transformer models, it requires specific tricks not easy to come with.  


> Want to understand how it works under the hood?  
> read [📕 Hugging Face Transformer inference UNDER 1 millisecond latency 📖](https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c?source=friends_link&sk=cd880e05c501c7880f2b9454830b8915)  
> <img src="resources/rabbit.jpg" width="120">

## Installation

<details><summary>Required dependencies</summary>

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

> to be able to leverage your CUDA installation by the Pycuda dependency, 
> don't forget to add to your $PATH env variable the path to CUDA. Otherwise, Pycuda will fail to compile.

</details>

```shell
git clone git@github.com:ELS-RD/transformer-deploy.git
cd transformer-deploy
pip3 install ".[GPU]" -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

To build your own version of the Docker image:

```shell
make build_docker
```

## Run in a single command

With the single command below, you will:

* **download** the model and its tokenizer from Hugging Face hub, 
* **convert** the model to ONNX graph,
* **optimize** 
  * the model with ONNX Runtime and save artefact (`model.onnx`),
  * the model with TensorRT and save artefact (`model.plan`),
* **benchmark** each backend (including Pytorch),
* **generate** configuration files for Triton inference server

```shell
convert_model -m roberta-large-mnli --backend tensorrt onnx --seq-len 16 128 128 --batch-size 1 32 32
```

> **16 128 128** -> minimum, optimal, maximum sequence length, to help TensorRT better optimize your model  
> **1 32 32** -> batch size, same as above

* Launch Nvidia Triton inference server to play with both ONNX and TensorRT models:

```shell
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:21.11-py3 \
  bash -c "pip install transformers sentencepiece && tritonserver --model-repository=/models"
```

> As you can see we install Transformers and then launch the server itself.  
> This is of course a bad practice, you should make your own 2 lines Dockerfile with Transformers inside.

Right now, only TensorRT 8.0.3 backend is available in Triton.  
Until the TensorRT 8.2 backend is available, we advise you to only use ONNX Runtime Triton backend.   

* Query the inference server:

```shell
# https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md
# @ means no data conversion (curl feature)
curl -X POST  http://localhost:8000/v2/models/transformer_onnx_inference/versions/1/infer \
  --data-binary "@demo/query_body.bin" \
  --header "Inference-Header-Content-Length: 161"
```

> check [`demo`](./demo) folder to discover more performant ways to query the server from Python or elsewhere.

### TensorRT usage in Python script

If you just want to perform inference inside your Python script (without any server) and still get the best TensorRT performance, check:

* [convert.py](./src/transformer_deploy/convert.py)
* [trt_utils.py](./src/transformer_deploy/backends/trt_utils.py)

#### High level explanations

* call `load_engine()` to parse an existing TensorRT engine
* setup a stream (for async call), a TensorRT runtime and a context
* load your profile(s)
* call `infer_tensorrt()`

... and you are done! 🎉

> if you are looking for inspiration, check [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)

## Benchmarks

Most transformer encoder based models are supported like Bert, Roberta, miniLM, Camembert, Albert, XLM-R, Distilbert, etc.  
**Best results are obtained with TensorRT 8.2.**  
Below examples are representative of the performance gain to expect from this library.  
Other improvements not shown here include GPU memory usage decrease, multi stream, etc.

### Small architecture

<details><summary>batch 1, seq length 16 on T4/RTX 3090 GPUs (up to 10X faster with TensorRT vs Pytorch)</summary>

command:

```shell
convert_model -m philschmid/MiniLM-L6-H384-uncased-sst2 --backend tensorrt onnx pytorch --seq-len 16 16 16 --batch-size 1 1 1
```

#### GPU Nvidia T4

```log
Inference done on Tesla T4
latencies:
[TensorRT (FP16)] mean=0.65ms, sd=0.11ms, min=0.57ms, max=0.96ms, median=0.59ms, 95p=0.93ms, 99p=0.94ms
[ONNX Runtime (vanilla)] mean=1.31ms, sd=0.05ms, min=1.27ms, max=1.48ms, median=1.30ms, 95p=1.44ms, 99p=1.45ms
[ONNX Runtime (optimized)] mean=0.71ms, sd=0.01ms, min=0.69ms, max=0.74ms, median=0.70ms, 95p=0.73ms, 99p=0.74ms
[Pytorch (FP32)] mean=5.01ms, sd=0.06ms, min=4.94ms, max=6.72ms, median=5.01ms, 95p=5.07ms, 99p=5.13ms
[Pytorch (FP16)] mean=5.44ms, sd=0.07ms, min=5.36ms, max=6.80ms, median=5.43ms, 95p=5.49ms, 99p=5.55ms
```

#### GPU Nvidia RTX 3090

```log
Inference done on NVIDIA GeForce RTX 3090
latencies:
[TensorRT (FP16)] mean=0.45ms, sd=0.05ms, min=0.41ms, max=0.78ms, median=0.45ms, 95p=0.55ms, 99p=0.73ms
[ONNX Runtime (vanilla)] mean=1.32ms, sd=0.11ms, min=1.24ms, max=2.36ms, median=1.30ms, 95p=1.50ms, 99p=1.74ms
[ONNX Runtime (optimized)] mean=0.84ms, sd=0.11ms, min=0.76ms, max=2.03ms, median=0.81ms, 95p=1.10ms, 99p=1.25ms
[Pytorch (FP32)] mean=4.68ms, sd=0.28ms, min=4.38ms, max=7.83ms, median=4.65ms, 95p=4.97ms, 99p=6.16ms
[Pytorch (FP16)] mean=5.25ms, sd=0.60ms, min=4.83ms, max=8.54ms, median=5.03ms, 95p=6.54ms, 99p=7.77ms
```

</details>

<details><summary>batch 16, seq length 384 on T4/RTX 3090 GPUs (up to 5X faster with TensorRT vs Pytorch)</summary>

command:

```shell
convert_model -m philschmid/MiniLM-L6-H384-uncased-sst2 --backend tensorrt onnx pytorch --seq-len 384 384 384 --batch-size 16 16 16
```

#### GPU Nvidia T4

```log
Inference done on Tesla T4
latencies:
[TensorRT (FP16)] mean=16.38ms, sd=0.30ms, min=15.45ms, max=17.42ms, median=16.42ms, 95p=16.83ms, 99p=17.09ms
[ONNX Runtime (vanilla)] mean=65.12ms, sd=1.53ms, min=61.74ms, max=68.51ms, median=65.21ms, 95p=67.46ms, 99p=67.90ms
[ONNX Runtime (optimized)] mean=26.75ms, sd=0.30ms, min=25.96ms, max=27.71ms, median=26.73ms, 95p=27.23ms, 99p=27.52ms
[Pytorch (FP32)] mean=82.22ms, sd=1.02ms, min=78.83ms, max=85.02ms, median=82.28ms, 95p=83.80ms, 99p=84.43ms
[Pytorch (FP16)] mean=46.29ms, sd=0.41ms, min=45.23ms, max=47.56ms, median=46.30ms, 95p=46.98ms, 99p=47.37ms
```

#### GPU Nvidia RTX 3090

```log
Inference done on NVIDIA GeForce RTX 3090
latencies:
[TensorRT (FP16)] mean=5.44ms, sd=0.45ms, min=5.03ms, max=8.91ms, median=5.20ms, 95p=6.11ms, 99p=7.39ms
[ONNX Runtime (vanilla)] mean=16.87ms, sd=2.15ms, min=15.38ms, max=26.03ms, median=15.82ms, 95p=22.63ms, 99p=24.20ms
[ONNX Runtime (optimized)] mean=8.07ms, sd=0.58ms, min=7.59ms, max=13.63ms, median=7.93ms, 95p=8.71ms, 99p=11.45ms
[Pytorch (FP32)] mean=17.09ms, sd=0.21ms, min=16.87ms, max=18.99ms, median=17.04ms, 95p=17.49ms, 99p=18.08ms
[Pytorch (FP16)] mean=14.77ms, sd=1.83ms, min=13.50ms, max=20.97ms, median=13.87ms, 95p=19.15ms, 99p=20.01ms
```

</details>

### Base architecture

<details><summary>batch 16, seq length 384 on T4/RTX 3090 GPUs (up to 5X faster with TensorRT vs Pytorch)</summary>

command:

```shell
convert_model -m cardiffnlp/twitter-roberta-base-sentiment --backend tensorrt onnx pytorch --seq-len 384 384 384 --batch-size 16 16 16
```

#### GPU Nvidia T4

```log
Inference done on Tesla T4
latencies:
[TensorRT (FP16)] mean=80.57ms, sd=1.00ms, min=76.23ms, max=83.16ms, median=80.53ms, 95p=82.14ms, 99p=82.53ms
[ONNX Runtime (vanilla)] mean=353.81ms, sd=14.79ms, min=335.54ms, max=390.86ms, median=348.41ms, 95p=382.09ms, 99p=386.84ms
[ONNX Runtime (optimized)] mean=97.94ms, sd=1.66ms, min=93.83ms, max=102.11ms, median=97.84ms, 95p=100.73ms, 99p=101.57ms
[Pytorch (FP32)] mean=398.49ms, sd=25.76ms, min=369.81ms, max=454.55ms, median=387.17ms, 95p=445.52ms, 99p=450.81ms
[Pytorch (FP16)] mean=134.18ms, sd=1.16ms, min=131.60ms, max=138.48ms, median=133.80ms, 95p=136.57ms, 99p=137.39ms
```

#### GPU Nvidia RTX 3090

```log
Inference done on NVIDIA GeForce RTX 3090
latencies:
[TensorRT (FP16)] mean=27.52ms, sd=1.61ms, min=24.49ms, max=33.78ms, median=28.01ms, 95p=30.33ms, 99p=31.22ms
[ONNX Runtime (vanilla)] mean=65.95ms, sd=6.18ms, min=60.84ms, max=99.75ms, median=62.97ms, 95p=81.02ms, 99p=89.10ms
[ONNX Runtime (optimized)] mean=32.73ms, sd=4.80ms, min=28.84ms, max=48.84ms, median=30.15ms, 95p=43.03ms, 99p=44.78ms
[Pytorch (FP32)] mean=69.18ms, sd=4.79ms, min=65.97ms, max=97.74ms, median=67.16ms, 95p=77.88ms, 99p=92.43ms
[Pytorch (FP16)] mean=48.78ms, sd=2.02ms, min=47.02ms, max=61.37ms, median=47.67ms, 95p=52.34ms, 99p=55.56ms
```

</details>

### Large architecture

<details><summary>batch 16, seq length 384 on T4/RTX 3090 GPUs (up to 5X faster with TensorRT vs Pytorch)</summary>

command:

```shell
convert_model -m roberta-large-mnli --backend tensorrt onnx pytorch --seq-len 384 384 384 --batch-size 16 16 16
```

#### GPU Nvidia T4

```log
Inference done on Tesla T4
latencies:
[TensorRT (FP16)] mean=240.39ms, sd=11.01ms, min=217.59ms, max=259.57ms, median=242.68ms, 95p=255.03ms, 99p=257.04ms
[ONNX Runtime (vanilla)] mean=1176.73ms, sd=63.51ms, min=1020.00ms, max=1225.03ms, median=1210.08ms, 95p=1217.54ms, 99p=1220.25ms
[ONNX Runtime (optimized)] mean=295.03ms, sd=19.69ms, min=255.74ms, max=314.78ms, median=307.07ms, 95p=311.20ms, 99p=312.47ms
[Pytorch (FP32)] mean=1220.41ms, sd=75.93ms, min=1119.93ms, max=1342.10ms, median=1216.23ms, 95p=1329.08ms, 99p=1336.47ms
[Pytorch (FP16)] mean=438.26ms, sd=13.71ms, min=398.29ms, max=459.97ms, median=442.36ms, 95p=453.96ms, 99p=457.57ms
```

#### GPU Nvidia RTX 3090

```log
Inference done on NVIDIA GeForce RTX 3090
latencies:
[TensorRT (FP16)] mean=79.54ms, sd=5.99ms, min=74.47ms, max=113.25ms, median=76.87ms, 95p=88.02ms, 99p=104.48ms
[ONNX Runtime (vanilla)] mean=202.88ms, sd=16.21ms, min=187.91ms, max=277.85ms, median=194.80ms, 95p=239.58ms, 99p=261.44ms
[ONNX Runtime (optimized)] mean=97.04ms, sd=5.55ms, min=90.83ms, max=121.88ms, median=94.04ms, 95p=104.81ms, 99p=107.75ms
[Pytorch (FP32)] mean=202.80ms, sd=11.16ms, min=194.47ms, max=284.70ms, median=198.46ms, 95p=221.72ms, 99p=257.31ms
[Pytorch (FP16)] mean=142.63ms, sd=6.35ms, min=136.24ms, max=189.95ms, median=139.90ms, 95p=154.10ms, 99p=160.16ms
```

</details>
