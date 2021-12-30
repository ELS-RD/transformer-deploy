# TensorRT usage in Python script

If you just want to perform inference inside your Python script (without any server) and still get the best TensorRT performance, check:

* [convert.py](./src/transformer_deploy/convert.py)
* [trt_utils.py](./src/transformer_deploy/backends/trt_utils.py)

## High level explanations

* call `load_engine()` to parse an existing TensorRT engine
* setup a stream (for async call), a TensorRT runtime and a context
* load your profile(s)
* call `infer_tensorrt()`

... and you are done! 🎉

> if you are looking for inspiration, check [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)
