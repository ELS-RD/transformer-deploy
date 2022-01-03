# TensorRT usage in Python script

There are 2 ways to use a TensorRT optimized model:

* deploy it on Triton server
* use it directly in Python

This document is about the second option.

## High level explanations

* call `load_engine()` to parse an existing TensorRT engine or `build_engine()` to convert an ONNX file
* setup a CUDA `stream` (for async call), a TensorRT `runtime` and a `context`
* load your `profile`(s)
* call `infer_tensorrt()`

## Build engine

We assume that you have already prepared your ONNX file.  
Now we need to convert to TensorRT:

```python
import tensorrt as trt
from tensorrt.tensorrt import Logger, Runtime

from transformer_deploy.backends.trt_utils import build_engine

trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
runtime: Runtime = trt.Runtime(trt_logger)
profile_index = 0
max_seq_len = 256
batch_size = 32

engine = build_engine(
    runtime=runtime,
    onnx_file_path="model_qat.onnx",
    logger=trt_logger,
    min_shape=(1, max_seq_len),
    optimal_shape=(batch_size, max_seq_len),
    max_shape=(batch_size, max_seq_len),
    workspace_size=10000 * 1024 * 1024,
    fp16=True,
    int8=True,
)
```

## Prepare inference

Now the engine is ready, we can prepare the inference:

```python
import pycuda.autoinit
from pycuda._driver import Stream
from tensorrt.tensorrt import IExecutionContext

from transformer_deploy.backends.trt_utils import get_binding_idxs

stream: Stream = pycuda.driver.Stream()
context: IExecutionContext = engine.create_execution_context()
context.set_optimization_profile_async(profile_index=profile_index, stream_handle=stream.handle)
input_binding_idxs, output_binding_idxs = get_binding_idxs(engine, profile_index)  # type: List[int], List[int]
```

## Inference

```python


from transformer_deploy.backends.trt_utils import infer_tensorrt

input_np = ...

tensorrt_output = infer_tensorrt(
    context=context,
    host_inputs=input_np,
    input_binding_idxs=input_binding_idxs,
    output_binding_idxs=output_binding_idxs,
    stream=stream,
)
print(tensorrt_output)
```

... and you are done! 🎉

!!! tip

    To go deeper, check in the API:

    * `Convert`
    * `Backends/Trt utils`

    ... and if you are looking for inspiration, check [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)
