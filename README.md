# Submillisecond Transformer inference

Yes, you can perfom inference with transformer based model in less than 1ms on the cheapest GPU available on Amazon (T4)!

## Performance target

[Huggingface demo video](LINK)

* hardware: T4 / g4.dnn
* model: 
* experience 1 : seq len 16 tokens -> 1.7ms
* experience 2 : seq len 128 tokens -> 2.7ms

## Install dependencies

```shell
pip install -r requirements.txt
```

## Generate optimized models

```shell
DOCKER_BUILDKIT=1 docker build --tag onnxruntime-trt:latest -f Dockerfile .
docker run -it --rm --gpus all -v $PWD:/project onnxruntime-trt bash
# inside docker container
python /project/convert_onnx.py
```

## FastAPI server

```shell
# launch server, disable logging for best performances
uvicorn --log-level warning src.okdoc.server_onnx:app --port 8000 --host 0.0.0.0
# gunicorn -w 12 -k uvicorn.workers.UvicornWorker --log-level warning src.okdoc.server_onnx:app --bind 0.0.0.0:8000
# simple inference timing
time curl -G --data-urlencode query="This live event is great. I will sign-up for Infinity." localhost:8000/predict
# serious measures
sudo perf stat -r 10 -d curl -G --data-urlencode query="This live event is great. I will sign-up for Infinity." localhost:8000/predict
```

## Triton server

https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md

```shell
docker run -it --rm  -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:21.10-py3
pip install transformers # just to get tokenizer support
tritonserver --model-repository=/models
```

## Perf analysis

[documentation](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md)

```shell
# sudo apt install libb64-dev
perf_analyzer -m transformers --percentile=95 --input-data perf_data.json --shape TEXT:1 # -i grpc -a
```

# python -m onnxruntime.tools.symbolic_shape_infer --input output/okdoc-base-onnx/model.onnx --output output/okdoc-base-onnx/model2.onnx --auto_merge
# /usr/src/tensorrt/bin/trtexec --onnx="output/okdoc-base-onnx/qdq_model.onnx" --int8 --calib="/home/geantvert/workspace/okdoc/calibration.json" --workspace=6000
# /usr/src/tensorrt/bin/trtexec --onnx="output/okdoc-base-onnx/model.onnx" --shapes=input_ids:1x128,attention_mask:1x128 --workspace=6000 --best
