import tritonclient.http
import numpy as np
from utils import print_timings, setup_logging, track_infer_time

setup_logging()
model_name = 'sts'
url = '127.0.0.1:8000'
model_version = '1'

nb_tokens = 16
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)

time_buffer = list()
input0 = tritonclient.http.InferInput('input_ids', (1, nb_tokens), 'INT64')
input1 = tritonclient.http.InferInput('attention_mask', (1, nb_tokens), 'INT64')
input1.set_data_from_numpy(np.ones((1, nb_tokens), dtype=np.int64), binary_data=False)
output = tritonclient.http.InferRequestedOutput('model_output', binary_data=False)
for _ in range(10000):
    # random input to avoid any caching
    input0.set_data_from_numpy(np.random.randint(10000, size=(1, nb_tokens), dtype=np.int64), binary_data=False)
    _ = triton_client.infer(model_name, model_version=model_version, inputs=[input0, input1], outputs=[output])

for _ in range(1000):
    with track_infer_time(time_buffer):
        # random input to avoid any caching
        input0.set_data_from_numpy(np.random.randint(10000, size=(1, nb_tokens), dtype=np.int64), binary_data=False)
        response = triton_client.infer(model_name, model_version=model_version, inputs=[input0, input1],
                                       outputs=[output])
        logits = response.as_numpy('model_output')

print(logits)
print_timings(name="triton (onnx backend)", timings=time_buffer)
