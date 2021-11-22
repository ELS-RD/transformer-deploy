# Demo scripts

This folder contains scripts to run different benchmarks:

* `triton_client.py`: query the model with a string
* `triton_client_model.py`: query the model directly (without using the tokenizer) with numpy arrays
* `triton_client_requests.py`: query the model directly (without using the tokenizer) with numpy arrays using only `requests` library
* `triton_client_tokenizer.py`: query the tokenizer only
* `fast_api_server_onnx.py`: FastAPI inference server to compare to Nvidia Triton

Those examples may help you to implement the querying part in your application.
