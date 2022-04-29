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

"""
All the tooling to ease ONNX Runtime usage.
"""

import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cupy as cp
import numpy as np
import torch
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel


def create_model_for_provider(
    path: str, provider_to_use: Union[str, List], nb_threads: int = multiprocessing.cpu_count(), nb_instances: int = 0
) -> InferenceSession:
    """
    Create an ONNX Runtime instance.
    :param path: path to ONNX file or serialized to string model
    :param provider_to_use: provider to use for inference
    :param nb_threads: intra_op_num_threads to use
    :param nb_instances: inter_op_num_threads to use
    :return: ONNX Runtime inference session
    """
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if isinstance(provider_to_use, str):
        provider_to_use = [provider_to_use]
    if provider_to_use == ["CPUExecutionProvider"]:
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL if nb_instances <= 1 else ExecutionMode.ORT_PARALLEL
        options.intra_op_num_threads = nb_threads
        if nb_instances > 1:
            options.inter_op_num_threads = nb_instances
    options.inter_op_num_threads = 6
    options.intra_op_num_threads = 6
    return InferenceSession(path, options, providers=provider_to_use)


def optimize_onnx(
    onnx_path: str,
    onnx_optim_model_path: str,
    fp16: bool,
    use_cuda: bool,
    num_attention_heads: int = 0,
    hidden_size: int = 0,
    architecture: str = "bert",
) -> None:
    """
    ONNX Runtime transformer graph optimization.
    Performs some operator fusion (merge several nodes of the graph in a single one)
    and may convert some nodes to reduced precision.
    :param onnx_path: ONNX input path
    :param onnx_optim_model_path: where to save optimized model
    :param fp16: use mixed precision (faster inference)
    :param use_cuda: perform optimization on GPU (should )
    :param num_attention_heads: number of attention heads of a model (0 -> try to detect)
    :param hidden_size: hidden layer size of a model (0 -> try to detect)
    :param architecture: model architecture to optimize. One of [bert, bart, gpt2]
    """
    assert architecture in ["bert", "bart", "gpt2"], f"unsupported architecture: {architecture}"
    opt_level = 1 if architecture == "bert" else 0
    optimization_options = FusionOptions(model_type=architecture)
    optimization_options.enable_gelu_approximation = False  # additional optimization
    optimized_model: BertOnnxModel = optimizer.optimize_model(
        input=onnx_path,
        model_type=architecture,
        use_gpu=use_cuda,
        opt_level=opt_level,
        num_heads=num_attention_heads,  # automatic detection with 0 may not work with opset 13 or distilbert models
        hidden_size=hidden_size,  # automatic detection with 0
        optimization_options=optimization_options,
    )
    if fp16:
        # use_symbolic_shape_infer set to false because doesn't work after ONNX package v1.10.2
        optimized_model.convert_float_to_float16(use_symbolic_shape_infer=False)  # FP32 -> FP16
    logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(onnx_optim_model_path)


def cpu_quantization(input_model_path: str, output_model_path: str) -> None:
    """
    ONNX CPU only dynamic quantization.

    :param input_model_path: ONNX graph (float) to quantize
    :param output_model_path: where to save quantized model
    """
    quantize_dynamic(
        model_input=Path(input_model_path),
        model_output=Path(output_model_path),
        op_types_to_quantize=["MatMul", "Attention"],
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        extra_options={"WeightSymmetric": False, "MatMulConstBOnly": True},
    )


# https://github.com/pytorch/pytorch/blob/ac79c874cefee2f8bc1605eed9a924d80c0b3542/torch/testing/_internal/common_utils.py#L349
numpy_to_torch_dtype_dict = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}


# TODO add test + documentation
def to_pytorch(ort_tensor: OrtValue, np_type: type) -> torch.Tensor:
    if ort_tensor.device_name().lower() == "cuda":
        fake_owner = 1
        # size not used anywhere, so just put 0
        memory = cp.cuda.UnownedMemory(ort_tensor.data_ptr(), 0, fake_owner)
        memory_ptr = cp.cuda.MemoryPointer(memory, 0)
        # make sure you interpret the array shape/dtype/strides correctly
        cp_array = cp.ndarray(shape=ort_tensor.shape(), memptr=memory_ptr, dtype=np_type)
        return torch.from_dlpack(cp_array.toDlpack())
    else:
        np_tensor = ort_tensor.numpy()
        return torch.from_numpy(np_tensor)


def inference_onnx_binding(
    model_onnx: InferenceSession,
    inputs: Dict[str, torch.Tensor],
    device: str,
    output_shape: Any = None,  # TODO delete this parameter
    device_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Performs inference on ONNX Runtime in an optimized way.
    In particular, avoid some tensor copy from GPU to host by using Torch tensors directly.

    :param model_onnx: ONNX model
    :param inputs: input torch tensor
    :param device: where to run the inference. One of [cpu, cuda]
    :param output_shape: tensor output shape if known, otherwise will be guessed from axis names
    :param device_id: ID of the device where to run the inference, to be used when there are multiple GPUs, etc.
    :return: a dict {axis name: output tensor}
    """
    assert device in ["cpu", "cuda"]
    binding: IOBinding = model_onnx.io_binding()
    for input_onnx in model_onnx.get_inputs():
        if input_onnx.name not in inputs:  # some inputs may be optional
            continue
        tensor: torch.Tensor = inputs[input_onnx.name]
        tensor = tensor.detach()
        if tensor.dtype in [torch.int64, torch.long]:
            # int32 mandatory as input of bindings, int64 not supported
            tensor = tensor.type(dtype=torch.int32)
        tensor = tensor.contiguous()
        binding.bind_input(
            name=input_onnx.name,
            device_type=device,
            device_id=device_id,
            element_type=torch_to_numpy_dtype_dict[tensor.dtype],
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
        inputs[input_onnx.name] = tensor

    for out in model_onnx.get_outputs():
        binding.bind_output(
            name=out.name,
            device_type=device,
            device_id=device_id,
        )
    binding.synchronize_inputs()
    model_onnx.run_with_iobinding(binding)
    # WARNING: output type is hard coded
    outputs = {
        out.name: to_pytorch(t, np_type=np.float32) for out, t in zip(model_onnx.get_outputs(), binding.get_outputs())
    }
    return outputs
