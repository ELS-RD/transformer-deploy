import json
import struct


text: str = "My name is Wolfgang and I live in Berlin"

context_text: bytes = text.encode("UTF-8")

context_text_struct: bytes = struct.pack("<I", len(context_text)) + context_text

len_context_text_struct = len(context_text_struct)

data_struct = context_text_struct

request_data = {
    "inputs": [
        {
            "name": "TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "parameters": {"binary_data_size": len_context_text_struct},
        },
    ],
    "outputs": [{"name": "OUTPUT_TEXT", "parameters": {"binary_data": False}}],
}

data = json.dumps(request_data).encode() + data_struct

print(data)


with open("t5_query_body.bin", "wb") as f:
    f.write(data)


curl = f"""
curl -X POST  http://localhost:8000/v2/models/t5-dec-if-node_onnx_generate/versions/1/infer \
  --data-binary "@demo/generative-model/t5_query_body.bin" \
  --header "Inference-Header-Content-Length: {len(json.dumps(request_data).encode())}"
"""
print(curl)
import requests


res = requests.post(
    url="http://localhost:8000/v2/models/t5-dec-if-node_onnx_generate/versions/1/infer",
    data="@demo/generative-model/t5_query_body.bin",
    headers={"Inference-Header-Content-Length": len(json.dumps(request_data).encode()).to_bytes(5, "little")},
)
