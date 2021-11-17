import json
import zlib

import requests
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType

from benchmarks.utils import print_timings, setup_logging, track_infer_time


setup_logging()
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("philschmid/MiniLM-L6-H384-uncased-sst2")

tokens = tokenizer(
    text="This live event is great. I will sign-up for Infinity.",
    max_length=16,
    truncation=True,
    return_tensors=TensorType.NUMPY,
)

# https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md
url = "http://127.0.0.1:8000/v2/models/sts_model/versions/1/infer"
message = {
    "id": "42",
    "inputs": [
        {
            "name": "input_ids",
            "shape": tokens["input_ids"].shape,
            "datatype": "INT64",
            "data": tokens["input_ids"].tolist(),
        },
        {
            "name": "token_type_ids",
            "shape": tokens["token_type_ids"].shape,
            "datatype": "INT64",
            "data": tokens["token_type_ids"].tolist(),
        },
        {
            "name": "attention_mask",
            "shape": tokens["attention_mask"].shape,
            "datatype": "INT64",
            "data": tokens["attention_mask"].tolist(),
        },
    ],
    "outputs": [
        {
            "name": "output",
            "parameters": {"binary_data": False},
        }
    ],
}

time_buffer = list()
session = requests.Session()
for _ in range(10000):
    bytes_message = bytes(json.dumps(message), encoding="raw_unicode_escape")
    request_body = zlib.compress(bytes_message)
    _ = session.post(
        url,
        data=request_body,
        headers={
            "Content-Encoding": "gzip",
            "Accept-Encoding": "gzip",
            "Inference-Header-Content-Length": str(len(bytes_message)),
        },
    )

for _ in range(100):
    with track_infer_time(time_buffer):
        bytes_message = bytes(json.dumps(message), encoding="raw_unicode_escape")
        request_body = zlib.compress(bytes_message)
        _ = session.post(
            url,
            data=request_body,
            headers={
                "Content-Encoding": "gzip",
                "Accept-Encoding": "gzip",
                "Inference-Header-Content-Length": str(len(bytes_message)),
            },
        )

print_timings(name="triton (onnx backend) - requests", timings=time_buffer)
