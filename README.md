# 🤗 Hugging Face Transformer submillisecond inference 🤯 and deployment on Nvidia Triton server

Yes, you can perfom inference with transformer based model in less than 1ms on the cheapest GPU available on Amazon (T4)!

The commands below have been tested on a AWS G4.dnn with `Deep Learning Base AMI (Ubuntu 18.04) Version 44.0`.
They may require some small adaptations to be run on a another Linux distribution.

You can find explanations on how it works in 
[Hugging Face Transformer inference UNDER 1 millisecond latency](https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c?source=friends_link&sk=cd880e05c501c7880f2b9454830b8915)


## Baseline set by Hugging Face Infinity demo

[Hugging Face infinity demo video](https://www.youtube.com/watch?v=jiftCAhOYQA)

* AWS virtual machine: `g4dn.xlarge` (T4 GPU)
* model: `"philschmid/MiniLM-L6-H384-uncased-sst2"` (Hugging Face hub URL)
* experience 1 : batch size 1, seq len 16 tokens -> `1.7ms`
* experience 2 : batch size 1, seq len 128 tokens -> `2.5ms`

## Install dependencies

Those dependencies have to be installed on the remote machine directly (no container).

```shell
git clone git@github.com:ELS-RD/triton_transformers.git
pip3 install -r requirements.txt
```

## Generate optimized models

We generate the models from a Docker image so we can also get measures for `TensorRT + ONNX Runtime`.

```shell
cd triton_transformers
DOCKER_BUILDKIT=1 docker build --tag onnxruntime-trt:latest -f Dockerfile .
docker run -it --rm --gpus all -v $PWD:/project onnxruntime-trt bash -c "cd /project && python convert_onnx.py"
```

> ⚠️**WARNING**⚠️: if you run the conversion *outside* Docker container, you may have very different timings, and TensorRT won't work

It should produce something like that:

```log
10/31/2021 11:35:08 INFO     inference done on Tesla T4
10/31/2021 11:35:08 INFO     timing [[TensorrtExecutionProvider] ./onnx_models/model-shape.onnx]: mean=0.61ms, sd=0.11ms, min=0.52ms, max=0.92ms, median=0.54ms, 95p=0.88ms, 99p=0.90ms
10/31/2021 11:35:08 INFO     timing [[CUDAExecutionProvider] ./onnx_models/model.onnx]: mean=1.10ms, sd=0.10ms, min=1.04ms, max=3.44ms, median=1.07ms, 95p=1.29ms, 99p=1.36ms
10/31/2021 11:35:08 INFO     timing [[CUDAExecutionProvider] ./onnx_models/model-optimized.onnx]: mean=0.63ms, sd=0.05ms, min=0.60ms, max=0.84ms, median=0.61ms, 95p=0.77ms, 99p=0.79ms
10/31/2021 11:35:08 INFO     timing [Pytorch_32]: mean=5.09ms, sd=0.16ms, min=4.88ms, max=6.11ms, median=5.07ms, 95p=5.28ms, 99p=5.35ms
10/31/2021 11:35:08 INFO     timing [Pytorch_FP16]: mean=6.04ms, sd=0.74ms, min=5.77ms, max=28.79ms, median=6.05ms, 95p=6.19ms, 99p=6.29ms
```

`TensorRT` and optimized `ONNX Runtime` provides very similar results on short sequences.
In the following steps, we will continue with ONNX Runtime model because the dynamic axis are easier to work with compared to TensorRT. 

> Docker build will is very slow on a G4, be patient...
> the docker image is only required for `TensorRT` support inside `ONNX Runtime` (and measure a difference, if any, with ONNX Runtime).

## FastAPI server

This is our baseline, easy to run, but not very performant.

```shell
# launch server, disable logging for best performances
python3 -m uvicorn --log-level warning server_onnx:app --port 8000 --host 0.0.0.0
# other variation, 1 worker per CPU for best latency (plus not a good idea to have several times the same model on a single GPU):
python3 -m gunicorn -w 1 -k uvicorn.workers.UvicornWorker --log-level warning server_onnx:app --bind 0.0.0.0:8000

# simple inference timing
time curl -G --data-urlencode query="This live event is great. I will sign-up for Infinity." localhost:8000/predict
# slightly more serious measure
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
sudo perf stat -r 50 -d curl -G --data-urlencode query="This live event is great. I will sign-up for Infinity." localhost:8000/predict -s > /dev/null
```

It should produce:

```shell
Performance counter stats for 'curl -G --data-urlencode query=This live event is great. I will sign-up for Infinity. localhost:8000/predict' (50 runs):

              6.14 msec task-clock                #    0.494 CPUs utilized            ( +-  0.59% )
                 3      context-switches          #    0.462 K/sec                    ( +-  1.84% )
                 0      cpu-migrations            #    0.000 K/sec                  
               577      page-faults               #    0.094 M/sec                    ( +-  0.06% )
   <not supported>      cycles                                                      
   <not supported>      instructions                                                
   <not supported>      branches                                                    
   <not supported>      branch-misses                                               
   <not supported>      L1-dcache-loads                                             
   <not supported>      L1-dcache-load-misses                                       
   <not supported>      LLC-loads                                                   
   <not supported>      LLC-load-misses                                             

         0.0124429 +- 0.0000547 seconds time elapsed  ( +-  0.44% )
```

## Triton server

We want to copy the ONNX model we have generated in the first step in this folder.
Then we launch the Triton image. As you can see we install Transformers and then launch the server itself.
This is of course a bad practice, you should make your own 2 lines Dockerfile with Transformers inside.

```shell
# copy the generated model to triton model folder
cp ./onnx_models/model-optimized.onnx ./triton_models/sts/1/model.onnx
# install transformers (and its tokenizer) and launch server in a single line, ugly but good enough for our demo
# --shm-size 256m -> to have several Python backend at the same time
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:21.10-py3 \
  bash -c "pip install transformers && tritonserver --model-repository=/models"
```

## Triton server perf analysis

You need to edit the source code to load the 16 or 128 token sequence (the text is already included).

* 16 tokens:
```shell
ubuntu@ip-172-31-31-84:~/triton_transformers$ python3 triton_transformers.py 
10/31/2021 12:09:34 INFO     timing [triton transformers]: mean=1.53ms, sd=0.06ms, min=1.48ms, max=1.78ms, median=1.51ms, 95p=1.66ms, 99p=1.74ms
[[-3.4355469  3.2753906]]
```

* 128 tokens:
```shell
ubuntu@ip-XXX:~/triton_transformers$ python3 triton_transformers.py 
10/31/2021 12:12:00 INFO     timing [triton transformers]: mean=1.96ms, sd=0.08ms, min=1.88ms, max=2.24ms, median=1.93ms, 95p=2.17ms, 99p=2.23ms
[[-3.4589844  3.3027344]]
```

There is also a more serious performance analysis tool called perf_analyzer (it will take care to check that measures are stable, etc.).
[documentation](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md)
The tool need to be run on Ubuntu >= 20.04 (and won't work on Ubuntu 18.04 used for the AWS official Ubuntu deep learning image):
It also make measures on torchserve and tensorflow.

```shell
# perf_analyzer needs this dependency
sudo apt install libb64-dev
# add -a for async measures, and -i grpc to use that protocol instead of http 
~/.local/bin/perf_analyzer -m transformers --percentile=95 --input-data perf_data.json --shape TEXT:1 # -i grpc -a
```

## Call Triton HTTP API directly

If you don't want to use the `tritonclient` API, you can call the Triton server those ways:

```shell
# if you like Python requests library
python3 triton_requests.py

# if you want generic HTTP template, the @ means no data conversion
curl -X POST  http://localhost:8000/v2/models/transformers/versions/1/infer \
  --data-binary "@query_body.bin" \
  --header "Inference-Header-Content-Length: 160"
```

## Use TensorRT model in Triton server (instead of ONNX)

To use TensorRT model instead of ONNX Runtime one:

* we need to convert the ONNX to TensorRT engine
* update the configuration, TensorRT takes `int32` as input instead of `int64`

```shell
# we use Docker container to guarantee the use of the right trtexec version (otherwise you will have a deserialization error)
# it's a bacic conversion, IRL you want to provide minimum, optimimum and maximum shape at least
# it may take a few minutes...
docker run -it --rm --gpus all -v $PWD/onnx_models:/models nvcr.io/nvidia/tritonserver:21.10-py3 \
    /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/model.onnx \
    --best \
    --shapes=input_ids:1x128,attention_mask:1x128 \
    --saveEngine="/models/model.plan" \
    --workspace=6000
# move to triton model folder
cp ./onnx_models/model.plan ./triton_models/sts/1/model.plan
```

You then need to update you config.pbtxt in STS and tokenizer folders, replace all `TYPE_INT64` tensor type by `TYPE_INT32`.
In STS configuraiton file, replace `platform: "onnxruntime_onnx"` by `platform: "tensorrt_plan"`
Finally convert the numpy tensors to int32 in the tokenizer python code, like below (notice the `astype()`):

```python
input_ids = pb_utils.Tensor("INPUT_IDS", tokens['input_ids'].astype(np.int32))
attention = pb_utils.Tensor("ATTENTION", tokens['attention_mask'].astype(np.int32))
```

And you are done!
