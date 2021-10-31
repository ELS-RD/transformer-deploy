import tritonclient.http
from transformers import AutoTokenizer, TensorType

from utils import print_timings, setup_logging, track_infer_time

setup_logging()
tokenizer = AutoTokenizer.from_pretrained("philschmid/MiniLM-L6-H384-uncased-sst2")
model_name = 'cross'
url = '127.0.0.1:8000'
model_version = '1'

text = "This live event is great. I will sign-up for Infinity."
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

time_buffer = list()
input0 = tritonclient.http.InferInput('input_ids', (1, 16), 'INT64')
input1 = tritonclient.http.InferInput('attention_mask', (1, 16), 'INT64')

for _ in range(1000):
    with track_infer_time(time_buffer):
        tokens = tokenizer(text=text,
                           max_length=16,
                           truncation=True,
                           return_token_type_ids=False,
                           return_tensors=TensorType.NUMPY,
                           )

        input0.set_data_from_numpy(tokens['input_ids'], binary_data=False)
        input1.set_data_from_numpy(tokens["attention_mask"], binary_data=False)
        output = tritonclient.http.InferRequestedOutput('model_output', binary_data=False)
        response = triton_client.infer(model_name, model_version=model_version, inputs=[input0, input1],
                                       outputs=[output])
        logits = response.as_numpy('model_output')

print(logits)
print_timings(name="triton (onnx backend)", timings=time_buffer)
