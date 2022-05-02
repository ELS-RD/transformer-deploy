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
import copy
import logging
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np
import onnx
import tensorrt as trt
import torch
from onnx import ModelProto, NodeProto
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from tensorrt import ILayer, INetworkDefinition, LayerType


try:
    # noinspection PyUnresolvedReferences
    import cupy as cp
except ImportError:
    pass


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
    # ENABLE_ALL is for CPU (NCHWc -> NCHW layout)
    # https://onnxruntime.ai/docs/performance/graph-optimizations.html#layout-optimizations
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
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


# TODO add test including different input and checking that tensor is not overriden
# + remove typing arg (replaced by a mapping between ORT types and cupy ones)
def to_pytorch(ort_tensor: OrtValue, np_type: type, clone_tensor: bool) -> torch.Tensor:
    """
    Convert OrtValue output by Onnx Runtime to Pytorch tensor.
    The process can be done in a zero copy way (depending of clone parameter).
    :param ort_tensor: output from Onnx Runtime
    :param np_type: type of the tensor (numpy types)
    :param clone_tensor Onnx Runtime owns the storage array and will write on the next inference.
        By cloning you guarantee that the data won't change.
    :return: Pytorch tensor
    """
    if ort_tensor.device_name().lower() == "cuda":
        fake_owner = 1
        # size not used anywhere, so just put 0
        memory = cp.cuda.UnownedMemory(ort_tensor.data_ptr(), 0, fake_owner)
        memory_ptr = cp.cuda.MemoryPointer(memory, 0)
        # make sure you interpret the array shape/dtype/strides correctly
        cp_array = cp.ndarray(shape=ort_tensor.shape(), memptr=memory_ptr, dtype=np_type)
        # cloning required otherwise ORT will recycle the storage array and put new values into it if new inf is done.
        torch_tensor = torch.from_dlpack(cp_array.toDlpack())
        if clone_tensor:
            torch_tensor = torch_tensor.clone()
        return torch_tensor
    else:
        np_tensor = ort_tensor.numpy()
        return torch.from_numpy(np_tensor)


def inference_onnx_binding(
    model_onnx: InferenceSession,
    inputs: Dict[str, torch.Tensor],
    device: str,
    device_id: int = 0,
    binding: Optional[IOBinding] = None,
    clone_tensor: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Performs inference on ONNX Runtime in an optimized way.
    In particular, it avoids any Onnx Runtime output tensor copy.
    It means that Onnx Runtime is still owner of the array, and it will overwrite its content if you do another
    inference. To avoid any issue, just set clone_tensor to True (default).
    For best performance and lowest memory footprint, if you know what you are doing, set clone_tensor to True.

    :param model_onnx: ONNX model
    :param inputs: input torch tensor
    :param device: where to run the inference. One of [cpu, cuda]
    :param device_id: ID of the device where to run the inference, to be used when there are multiple GPUs, etc.
    :param binding: previously generated binding IO, will be reset.
    :param clone_tensor: clone Pytorch tensor to avoid its content being overwritten by Onnx Runtime
        at the next inference call.
    :return: a dict {axis name: output tensor}
    """
    assert isinstance(device, str)
    assert device in ["cpu", "cuda"], f"unexpected inference device: '{device}'"
    if binding is None:
        binding: IOBinding = model_onnx.io_binding()
    else:
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
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
    binding.synchronize_outputs()
    # WARNING: output type is hard coded
    outputs = {
        out.name: to_pytorch(t, np_type=np.float32, clone_tensor=clone_tensor)
        for out, t in zip(model_onnx.get_outputs(), binding.get_outputs())
    }
    return outputs


def add_output_nodes(model: ModelProto) -> ModelProto:
    """
    Set each node as output node for debugging purpose.
    :param model: ONNX model in protobuf format
    :return: modified ONNX model
    """
    model = copy.deepcopy(model)
    output_nodes = list()
    for n in model.graph.node:
        for output_name in n.output:
            output_nodes.append(onnx.ValueInfoProto(name=output_name))
    model.graph.output.extend(output_nodes)
    return model


def get_all_outputs(model: ModelProto, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Join ouput values from ONNX model with their names.
    :param model: ONNX model
    :param inputs: model input
    :return: tensor outputs and their names
    """
    return inference_onnx_binding(
        model_onnx=model, inputs=inputs, device=list(inputs.values())[0].device.type, clone_tensor=False
    )


def find_node_fp32(graph: Dict[str, Set[str]], output_nodes: Dict[str, torch.Tensor]) -> List[str]:
    """
    Identify out of range values in model output.
    :param graph: graph as adjency nodes dict
    :param output_nodes: output of each node
    :return: list of nodes producing outputs outside fp16 tensor
    """
    keep_fp32 = list()
    for k, v in output_nodes.items():
        if v.dtype != torch.float32:
            continue
        np_v = v.detach().cpu().numpy()
        if np.max(np_v) > np.finfo(np.float16).max or np.min(np_v) < np.finfo(np.float16).min:
            keep_fp32 += [n for n in graph[k]]
    return keep_fp32


def get_fix_fp16_network_func(keep_fp32: List[str]) -> Callable[[INetworkDefinition], INetworkDefinition]:
    """
    Generate a function to set precision of specific nodes to FP32 to keep tensorrt FP16 output close to FP32 nodes
    :param keep_fp32: nodes to keep in FP32
    :return: a function to set node precisions
    """

    def f(network_definition: INetworkDefinition) -> INetworkDefinition:
        for layer_index in range(network_definition.num_layers - 1):
            layer: ILayer = network_definition.get_layer(layer_index)
            # next layer should take FP16 as input
            next_layer: ILayer = network_definition.get_layer(layer_index + 1)

            if layer.name in keep_fp32 and next_layer.type != LayerType.IDENTITY:
                layer.precision = trt.DataType.FLOAT
                layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
                # identity function is mainly used for casting
                # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Layers.html#iidentitylayer
                if next_layer.type != LayerType.IDENTITY:
                    next_layer.precision = trt.DataType.FLOAT
                    # next_layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)

        return network_definition

    return f


def get_adjency_dict(model: ModelProto) -> Dict[str, Set[str]]:
    """
    Convert ONNX model to adjency
    :param model: ONNX model
    :return: a dict of links from input to output nodes
    """
    adj_dict: Dict[str, Set[str]] = defaultdict(set)
    for n in model.graph.node:  # type: NodeProto
        assert len(n.output) == 1
        output_node = n.output[0]
        adj_dict[output_node].add(n.name)
    return adj_dict


def get_list_fp32_nodes(
    model: InferenceSession,
    onnx_graph: Dict[str, Set[str]],
    get_input: Callable[[], Dict[str, torch.Tensor]],
    nb_try: int,
) -> List[str]:
    """
    Find the list of nodes to keep in FP32 to avoid out of range values
    :param model: model to test
    :param onnx_graph: a dict of links from input to output nodes
    :param get_input: generate input to test the model. Output should change from call to call.
    :param nb_try: nb of tests to perform. More is better and slower
    :return: list of names of nodes to keep in FP32
    """
    keep_fp32_nodes = list()
    for _ in range(nb_try):
        outputs: Dict[str, torch.Tensor] = get_all_outputs(model=model, inputs=get_input())
        keep_node_io = find_node_fp32(graph=onnx_graph, output_nodes=outputs)
        keep_fp32_nodes.extend([n for n in keep_node_io if n not in keep_fp32_nodes])
    return keep_fp32_nodes
