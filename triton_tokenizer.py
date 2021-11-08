import tritonclient.http
import numpy as np
from utils import print_timings, setup_logging, track_infer_time

model_name = 'tokenize'
url = '127.0.0.1:8000'
model_version = '1'
text = "SOME TEXT"  # edit to check longer sequence length
batch_size = 1

setup_logging()
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
time_buffer = list()

query = tritonclient.http.InferInput(name='TEXT', shape=(batch_size,), datatype="BYTES")
input_ids = tritonclient.http.InferRequestedOutput('INPUT_IDS', binary_data=False)
attention = tritonclient.http.InferRequestedOutput('ATTENTION', binary_data=False)


def perform_inference():
    query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
    triton_client.infer(model_name, model_version=model_version, inputs=[query], outputs=[input_ids, attention])


# warmup
for _ in range(10000):
    perform_inference()

for _ in range(1000):
    with track_infer_time(time_buffer):
        perform_inference()

print_timings(name=f"tokenize, # text len: {len(text)}", timings=time_buffer)
