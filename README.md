# Submillisecond Transformer inference

Yes, you can perfom inference with transformer based model in less than 1ms on the cheapest GPU available on Amazon (T4)!

The command below have been tested on a AWS G4.dnn with IMAGE VERSION USED.
They may need some small adaptations to be run on a modern Linux

## Performance target

[Huggingface demo video](https://www.youtube.com/watch?v=jiftCAhOYQA)

* hardware: T4 / g4.dnn
* model: "philschmid/MiniLM-L6-H384-uncased-sst2"
* experience 1 : seq len 16 tokens -> 1.7ms
* experience 2 : seq len 128 tokens -> 2.7ms

## Install dependencies

To be installed on the remote machine directly (no container).

```shell
git clone git@github.com:ELS-RD/triton_transformers.git
cd triton_transformers
pip3 install -r requirements.txt
```

## Generate optimized models

```shell
# inside the triton_transformers folder
DOCKER_BUILDKIT=1 docker build --tag onnxruntime-trt:latest -f Dockerfile .
docker run -it --rm --gpus all -v $PWD:/project onnxruntime-trt bash -c cd /project && python convert_onnx.py
```

```log
10/31/2021 11:35:08 INFO     inference done on Tesla T4
10/31/2021 11:35:08 INFO     timing [[TensorrtExecutionProvider] ./onnx_models/model-shape.onnx]: mean=0.61ms, sd=0.11ms, min=0.52ms, max=0.92ms, median=0.54ms, 95p=0.88ms, 99p=0.90ms
10/31/2021 11:35:08 INFO     timing [[CUDAExecutionProvider] ./onnx_models/model.onnx]: mean=1.10ms, sd=0.10ms, min=1.04ms, max=3.44ms, median=1.07ms, 95p=1.29ms, 99p=1.36ms
10/31/2021 11:35:08 INFO     timing [[CUDAExecutionProvider] ./onnx_models/model-optimized.onnx]: mean=0.63ms, sd=0.05ms, min=0.60ms, max=0.84ms, median=0.61ms, 95p=0.77ms, 99p=0.79ms
10/31/2021 11:35:08 INFO     timing [Pytorch_32]: mean=5.09ms, sd=0.16ms, min=4.88ms, max=6.11ms, median=5.07ms, 95p=5.28ms, 99p=5.35ms
10/31/2021 11:35:08 INFO     timing [Pytorch_fp16]: mean=6.04ms, sd=0.74ms, min=5.77ms, max=28.79ms, median=6.05ms, 95p=6.19ms, 99p=6.29ms
```

TensorRT and optimized onnxruntime provides very similar results on short sequences.
In the following steps, we will continu with onnxruntime model because the dynamic axis are easier to work with compared to TensorRT. 

> Docker build will take almost 1 hour...
> the docker image is only required for `tensorrt` support inside `onnxruntime` (and measure a difference, if any, with onnxruntime).
> if you don't want to wait, you can also just install onnxruntime-gpu from pip on the machine (no container) 
> and disable the line `("TensorrtExecutionProvider", infered_shape_model_onnx_path),` to run the script directly on the machine

## FastAPI server

```shell
# launch server, disable logging for best performances
python3 -m uvicorn --log-level warning server_onnx:app --port 8000 --host 0.0.0.0
# other variation, 1 worker per CPU for best latency (plus not a good idea to have several times the same model on a single GPU):
python3 -m gunicorn -w 1 -k uvicorn.workers.UvicornWorker --log-level warning server_onnx:app --bind 0.0.0.0:8000

# simple inference timing
time curl -G --data-urlencode query="This live event is great. I will sign-up for Infinity." localhost:8000/predict
# serious measures
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
sudo perf stat -r 50 -d curl -G --data-urlencode query="This live event is great. I will sign-up for Infinity." localhost:8000/predict
```

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

https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md

```shell
# copy the generated model
cp ./onnx_models/model-optimized.onnx ./triton_models/cross/1/model.onnx
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:21.10-py3
# inside the container
pip install transformers # just to get tokenizer support
tritonserver --model-repository=/models # launch server
```

## Perf analysis

```shell
python3 triton_transformers.py
```

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

[documentation](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md)
To run on your machine only (the tool won't work on Ubuntu 18.04 :-( ):

```shell
# perf_analyzer needs this dependency
sudo apt install libb64-dev
# add -a for async measures, and -i grpc to use that protocol instead of http 
~/.local/bin/perf_analyzer -m transformers --percentile=95 --input-data perf_data.json --shape TEXT:1 # -i grpc -a
```

# python -m onnxruntime.tools.symbolic_shape_infer --input output/okdoc-base-onnx/model.onnx --output output/okdoc-base-onnx/model2.onnx --auto_merge
# /usr/src/tensorrt/bin/trtexec --onnx="output/okdoc-base-onnx/qdq_model.onnx" --int8 --calib="/home/geantvert/workspace/okdoc/calibration.json" --workspace=6000
# /usr/src/tensorrt/bin/trtexec --onnx="output/okdoc-base-onnx/model.onnx" --shapes=input_ids:1x128,attention_mask:1x128 --workspace=6000 --best
