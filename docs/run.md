## Command line

!!! tip

    You can run commands below from Docker directly (no need to install Nvidia dependencies outside Docker one), like:
    ```shell
    docker run -it --rm --gpus all \
    -v $PWD:/project ghcr.io/els-rd/transformer-deploy:latest \
    bash -c "cd /project && \
      convert_model -m \"philschmid/MiniLM-L6-H384-uncased-sst2\" \
      --backend onnx \
      --seq-len 128 128 128"
    ```

With the single command below, you will:

* **download** the model and its tokenizer from :hugging: Hugging Face hub, 
* **convert** the model to ONNX graph,
* **optimize** 
  * the model with ONNX Runtime and save artefact (`model.onnx`),
  * the model with TensorRT and save artefact (`model.plan`),
* **benchmark** each backend (including Pytorch),
* **generate** configuration files for Triton inference server

```shell
convert_model -m philschmid/MiniLM-L6-H384-uncased-sst2 --backend onnx --seq-len 128 128 128 --batch-size 1 32 32
# ...
# Inference done on NVIDIA GeForce RTX 3090
# latencies:
# [Pytorch (FP32)] mean=8.75ms, sd=0.30ms, min=8.60ms, max=11.20ms, median=8.68ms, 95p=9.15ms, 99p=10.77ms
# [Pytorch (FP16)] mean=6.75ms, sd=0.22ms, min=6.66ms, max=8.99ms, median=6.71ms, 95p=6.88ms, 99p=7.95ms
# [ONNX Runtime (FP32)] mean=8.10ms, sd=0.43ms, min=7.93ms, max=11.76ms, median=8.02ms, 95p=8.39ms, 99p=11.30ms
# [ONNX Runtime (optimized)] mean=3.66ms, sd=0.23ms, min=3.57ms, max=6.46ms, median=3.62ms, 95p=3.70ms, 99p=4.95ms
```

!!! info

    **128 128 128** -> minimum, optimal, maximum sequence length, to help TensorRT better optimize your model. 
    Better to have the same value for seq len to get best performances from TensorRT (ONNX Runtime has not this limitation).

    **1 32 32** -> batch size, same as above. Good idea to get 1 as minimum value. No impact on TensorRT performance.

* Launch Nvidia Triton inference server to play with both ONNX and TensorRT models:

```shell
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:22.01-py3 \
  bash -c "pip install transformers sentencepiece && tritonserver --model-repository=/models"
```

> As you can see, we install Transformers and then launch the server itself.  
> This is, of course, a bad practice, you should make your own 2 lines Dockerfile with Transformers inside.

## Query the inference server

To query your inference server you first need to need to convert your string to a binary file following the same format as in `demo/infinity/query_body.bin`. 

The `query_body.bin` file is composed of three parts, a header describing the inputs/outputs values, a binary value representing the length of the binary input data and the binary input data.

The header part is straightforward as it's simply a JSON describing the inputs and outputs values:
```json
{"inputs":[{"name":"TEXT","shape":[1],"datatype":"BYTES","parameters":{"binary_data_size":59}}],"outputs":[{"name":"output","parameters":{"binary_data":false}}]}
```
> Note that we provide the `binary_data_size` value describing the lenght of the content following the header.

The integer representing the length of the input data is written in little endian: `7`.
> Note that the length of the content is not 7 but 55 as the `7` in the 55th ascii character.

The binary input data is simply a text encoded in `UTF-8` format: `This live event is great. I will sign-up for Infinity.`
(If you have several strings, just concatenate the results.)

You can follow the code below for the recipe for a single string. It will also tell you the value to specify within the `Inference-Header-Content-Length` header.
```python
import struct

text_b: bytes = "This live event is great. I will sign-up for Infinity.\n".encode("UTF-8")

prefix: bytes = b'{"inputs":[{"name":"TEXT","shape":[1],"datatype":"BYTES","parameters":{"binary_data_size":' + bytes(str(len(text_b) + len(struct.pack("<I", len(text_b)))), encoding='utf8') + b'}}],"outputs":[{"name":"output","parameters":{"binary_data":false}}]}'

print('--header "Inference-Header-Content-Length: ', len(prefix), '"', sep="")

with open("body.bin", "wb+") as f:
  f.write(prefix + struct.pack("<I", len(text_b)) + text_b)
  # <I means little-endian unsigned integers, followed by the number of elements
```

!!! tip

    check Nvidia implementation from [https://github.com/triton-inference-server/client/blob/530bcac5f1574aa2222930076200544eb274245c/src/python/library/tritonclient/utils/__init__.py#L187](https://github.com/triton-inference-server/client/blob/530bcac5f1574aa2222930076200544eb274245c/src/python/library/tritonclient/utils/__init__.py#L187)
    for more information.

You can now query your model using your input file :
```shell
# https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md
# @ means no data conversion (curl feature)
curl -X POST  http://localhost:8000/v2/models/transformer_onnx_inference/versions/1/infer \
  --data-binary "@demo/infinity/query_body.bin" \
  --header "Inference-Header-Content-Length: 161"
```

> check [`demo`](https://github.com/ELS-RD/transformer-deploy/tree/main/demo/infinity) folder to discover more performant ways to query the server from Python or elsewhere.

--8<-- "resources/abbreviations.md"