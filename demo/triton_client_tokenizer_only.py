#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import tritonclient.http

from transformer_deploy.benchmarks.utils import print_timings, setup_logging, track_infer_time


model_name = "transformer_onnx_tokenize"
url = "127.0.0.1:8000"
model_version = "1"
text = "SOME TEXT"  # edit to check longer sequence length
batch_size = 1

setup_logging()
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
time_buffer = list()

query = tritonclient.http.InferInput(name="TEXT", shape=(batch_size,), datatype="BYTES")
input_ids = tritonclient.http.InferRequestedOutput("input_ids", binary_data=False)
attention_mask = tritonclient.http.InferRequestedOutput("attention_mask", binary_data=False)
token_type_ids = tritonclient.http.InferRequestedOutput("token_type_ids", binary_data=False)


def perform_inference():
    query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
    output = triton_client.infer(
        model_name, model_version=model_version, inputs=[query], outputs=[input_ids, attention_mask, token_type_ids]
    )
    return output.as_numpy("input_ids"), output.as_numpy("token_type_ids"), output.as_numpy("attention_mask")


# warmup
for _ in range(100):
    perform_inference()

for _ in range(1000):
    with track_infer_time(time_buffer):
        perform_inference()

print_timings(name=f"tokenize, # text len: {len(text)}", timings=time_buffer)
