import numpy as np
import tritonclient.http

from utils import setup_logging, track_infer_time, print_timings

setup_logging()
model_name = 'transformers'
url = '127.0.0.1:8000'
model_version = '1'

# from https://venturebeat.com/2021/08/25/how-hugging-face-is-tackling-bias-in-nlp/, text used in the HF demo
text_128 = """Today, Hugging Face has expanded to become a robust NLP startup, 
known primarily for making open-source software such as Transformers and Datasets, 
used for building NLP systems. “The software Hugging Face develops can be used for classification, question answering, 
translation, and many other NLP tasks,” Rush said. Hugging Face also hosts a range of pretrained NLP models, 
on GitHub, that practitioners can download and apply for their problems, Rush added."""
# comment line below to measure on 128 tokens
text_16 = "This live event is great. I will sign-up for Infinity."
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(model_name=model_name, model_version=model_version), "model not yet ready"

model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

query = tritonclient.http.InferInput(name='TEXT', shape=(1,), datatype="BYTES")
model_output = tritonclient.http.InferRequestedOutput(name='model_output', binary_data=False)
time_buffer = list()
for _ in range(10000):
    query.set_data_from_numpy(np.asarray([text_16], dtype=object))
    response = triton_client.infer(model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_output])
for _ in range(100):
    with track_infer_time(time_buffer):
        query.set_data_from_numpy(np.asarray([text_16], dtype=object))
        response = triton_client.infer(model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_output])


print_timings(name="triton transformers", timings=time_buffer)
print(response.as_numpy('model_output'))
