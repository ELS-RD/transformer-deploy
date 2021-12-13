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

from typing import List, OrderedDict, Tuple

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from numpy import ndarray
from pycuda._driver import DeviceAllocation, Stream
from tensorrt import ICudaEngine, IExecutionContext
from tensorrt.tensorrt import (
    Builder,
    IBuilderConfig,
    IElementWiseLayer,
    ILayer,
    INetworkDefinition,
    IOptimizationProfile,
    IReduceLayer,
    Logger,
    OnnxParser,
    Runtime,
)


class Calibrator(trt.IInt8Calibrator):
    def __init__(self):
        trt.IInt8Calibrator.__init__(self)
        self.algorithm = trt.CalibrationAlgoType.MINMAX_CALIBRATION
        self.batch_size = 32

        input_list: List[ndarray] = [np.zeros((32, 512), dtype=np.int32) for _ in range(3)]
        # allocate GPU memory for input tensors
        self.device_inputs: List[DeviceAllocation] = [cuda.mem_alloc(tensor.nbytes) for tensor in input_list]
        for h_input, d_input in zip(input_list, self.device_inputs):
            cuda.memcpy_htod_async(d_input, h_input)  # host to GPU
        self.count = 0

    def get_algorithm(self):
        return trt.CalibrationAlgoType.MINMAX_CALIBRATION

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        self.count += 1
        if self.count > 20:
            return []
        # return pointers to arrays
        return [int(d) for d in self.device_inputs]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        with open("calibration_cache.bin", "wb") as f:
            f.write(cache)

    def free(self):
        for dinput in self.device_inputs:
            dinput.free()


def setup_binding_shapes(
    context: trt.IExecutionContext,
    host_inputs: List[np.ndarray],
    input_binding_idxs: List[int],
    output_binding_idxs: List[int],
) -> Tuple[List[np.ndarray], List[DeviceAllocation]]:
    # explicitly set dynamic input shapes, so dynamic output shapes can be computed internally
    for host_input, binding_index in zip(host_inputs, input_binding_idxs):
        context.set_binding_shape(binding_index, host_input.shape)
    assert context.all_binding_shapes_specified
    host_outputs: List[np.ndarray] = []
    device_outputs: List[DeviceAllocation] = []
    for binding_index in output_binding_idxs:
        output_shape = context.get_binding_shape(binding_index)
        # allocate buffers to hold output results after copying back to host
        buffer = np.empty(output_shape, dtype=np.float32)
        host_outputs.append(buffer)
        # allocate output buffers on device
        device_outputs.append(cuda.mem_alloc(buffer.nbytes))
    return host_outputs, device_outputs


def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
    # calculate start/end binding indices for current context's profile
    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles_bindings
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile  # Separate input and output binding indices for convenience
    input_binding_idxs = []
    output_binding_idxs = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_idxs.append(binding_index)
        else:
            output_binding_idxs.append(binding_index)
    return input_binding_idxs, output_binding_idxs


def fix_fp16_network(network_definition: INetworkDefinition) -> INetworkDefinition:
    # search for patterns which may overflow in FP16 precision, we force FP32 precisions for those nodes
    for layer_index in range(network_definition.num_layers - 1):
        layer: ILayer = network_definition.get_layer(layer_index)
        next_layer: ILayer = network_definition.get_layer(layer_index + 1)
        # POW operation usually followed by mean reduce
        if layer.type == trt.LayerType.ELEMENTWISE and next_layer.type == trt.LayerType.REDUCE:
            # casting to get access to op attribute
            layer.__class__ = IElementWiseLayer
            next_layer.__class__ = IReduceLayer
            if layer.op == trt.ElementWiseOperation.POW:
                layer.precision = trt.DataType.FLOAT
                next_layer.precision = trt.DataType.FLOAT
            layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
            next_layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
    return network_definition


def build_engine(
    runtime: Runtime,
    onnx_file_path: str,
    logger: Logger,
    min_shape: Tuple[int, int],
    optimal_shape: Tuple[int, int],
    max_shape: Tuple[int, int],
    workspace_size: int,
    fp16: bool,
    int8: bool,
) -> ICudaEngine:
    with trt.Builder(logger) as builder:  # type: Builder
        with builder.create_network(
            flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network_definition:  # type: INetworkDefinition
            with trt.OnnxParser(network_definition, logger) as parser:  # type: OnnxParser
                builder.max_batch_size = max_shape[0]  # max batch size
                config: IBuilderConfig = builder.create_builder_config()
                config.max_workspace_size = workspace_size
                # to enable complete trt inspector debugging, only for TensorRT >= 8.2
                # config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
                # disable CUDNN optimizations
                config.set_tactic_sources(
                    tactic_sources=1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT)
                )
                config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
                if int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                if fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
                # https://github.com/NVIDIA/TensorRT/issues/1196 (sometimes big diff in output when using FP16)
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                with open(onnx_file_path, "rb") as f:
                    parser.parse(f.read())
                profile: IOptimizationProfile = builder.create_optimization_profile()
                for num_input in range(network_definition.num_inputs):
                    profile.set_shape(
                        input=network_definition.get_input(num_input).name,
                        min=min_shape,
                        opt=optimal_shape,
                        max=max_shape,
                    )
                config.add_optimization_profile(profile)
                if fp16:
                    network_definition = fix_fp16_network(network_definition)
                trt_engine = builder.build_serialized_network(network_definition, config)
                engine: ICudaEngine = runtime.deserialize_cuda_engine(trt_engine)
                assert engine is not None, "error during engine generation, check error messages above :-("
                return engine


def save_engine(engine: ICudaEngine, engine_file_path: str):
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())


def load_engine(runtime: Runtime, engine_file_path: str) -> ICudaEngine:
    with open(engine_file_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def infer_tensorrt(
    context: IExecutionContext,
    host_inputs: OrderedDict[str, np.ndarray],
    input_binding_idxs: List[int],
    output_binding_idxs: List[int],
    stream: Stream,
) -> np.ndarray:
    input_list: List[ndarray] = list()
    device_inputs: List[DeviceAllocation] = list()
    for tensor in host_inputs.values():
        # warning: small change in output if int64 is used instead of int32
        tensor_int32: np.ndarray = np.asarray(tensor, dtype=np.int32)
        input_list.append(tensor_int32)
        # allocate GPU memory for input tensors
        device_input: DeviceAllocation = cuda.mem_alloc(tensor_int32.nbytes)
        device_inputs.append(device_input)
        cuda.memcpy_htod_async(device_input, tensor_int32.ravel(), stream)
    # calculate input shape, bind it, allocate GPU memory for the output
    host_outputs, device_outputs = setup_binding_shapes(context, input_list, input_binding_idxs, output_binding_idxs)
    bindings = device_inputs + device_outputs
    assert context.execute_async_v2(bindings, stream_handle=stream.handle), "failure during execution of inference"
    for h_output, d_output in zip(host_outputs, device_outputs):
        cuda.memcpy_dtoh_async(h_output, d_output)  # GPU to host
    stream.synchronize()  # sync all CUDA ops
    return host_outputs
