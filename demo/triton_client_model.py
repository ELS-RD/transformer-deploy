#  Copyright 2021, Lefebvre Sarrut Services
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


model_name = "transformer_onnx_model"
url = "127.0.0.1:8000"
model_version = "1"
nb_tokens = 16  # edit to check longer sequence length

setup_logging()
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
time_buffer = list()

input0 = tritonclient.http.InferInput("input_ids", (1, nb_tokens), "INT64")
input1 = tritonclient.http.InferInput("attention_mask", (1, nb_tokens), "INT64")
input1.set_data_from_numpy(np.ones((1, nb_tokens), dtype=np.int64), binary_data=False)
input2 = tritonclient.http.InferInput("token_type_ids", (1, nb_tokens), "INT64")
input2.set_data_from_numpy(np.ones((1, nb_tokens), dtype=np.int64), binary_data=False)
output = tritonclient.http.InferRequestedOutput("output", binary_data=False)


def perform_random_inference():
    input0.set_data_from_numpy(np.random.randint(10000, size=(1, nb_tokens), dtype=np.int64), binary_data=False)
    triton_client.infer(model_name, model_version=model_version, inputs=[input0, input1, input2], outputs=[output])


# warmup
for _ in range(100):
    perform_random_inference()

for _ in range(1000):
    with track_infer_time(time_buffer):
        perform_random_inference()

print_timings(name=f"triton, # tokens: {nb_tokens}", timings=time_buffer)
