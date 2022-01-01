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

"""
All the tooling to ease ONNX Runtime usage.
"""

import logging
import multiprocessing
from collections import OrderedDict
from typing import List
from typing import OrderedDict as Od
from typing import Union

import torch
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from torch.onnx import TrainingMode
from transformers import PreTrainedModel


def create_model_for_provider(
    path: str, provider_to_use: Union[str, List], nb_threads: int = multiprocessing.cpu_count(), nb_instances: int = 0
) -> InferenceSession:
    """
    Create an ONNX Runtime instance.
    :param path: path to ONNX file
    :param provider_to_use: provider to use for inference
    :param nb_threads: intra_op_num_threads to use
    :param nb_instances: inter_op_num_threads to use
    :return: ONNX Runtime inference session
    """
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if type(provider_to_use) != list:
        provider_to_use = [provider_to_use]
    if provider_to_use == ["CPUExecutionProvider"]:
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL if nb_instances <= 1 else ExecutionMode.ORT_PARALLEL
        options.intra_op_num_threads = nb_threads
        if nb_instances > 1:
            options.inter_op_num_threads = nb_instances
    return InferenceSession(path, options, providers=provider_to_use)


def convert_to_onnx(
    model_pytorch: PreTrainedModel, output_path: str, inputs_pytorch: Od[str, torch.Tensor], opset: int = 12
) -> None:
    """
    Convert a Pytorch model to an ONNX graph by tracing the provided input inside the Pytorch code.
    :param model_pytorch: Pytorch model
    :param output_path: where to save ONNX file
    :param inputs_pytorch: Tensor, can be dummy data, shape is not important as we declare all axes as dynamic.
    Should be on the same device than the model (CPU or GPU)
    :param opset: version of ONNX protocol to use, usually 12, or 13 if you use per channel quantized model
    """
    # dynamic axis == variable length axis
    dynamic_axis = OrderedDict()
    for k in inputs_pytorch.keys():
        dynamic_axis[k] = {0: "batch_size", 1: "sequence"}
    dynamic_axis["output"] = {0: "batch_size"}
    with torch.no_grad():
        torch.onnx.export(
            model_pytorch,  # model to optimize
            args=tuple(inputs_pytorch.values()),  # tuple of multiple inputs
            f=output_path,  # output path / file object
            opset_version=opset,  # the ONNX version to use, 13 if quantized model, 12 for not quantized ones
            do_constant_folding=True,  # simplify model (replace constant expressions)
            input_names=list(inputs_pytorch.keys()),  # input names
            output_names=["output"],  # output axis name
            dynamic_axes=dynamic_axis,  # declare dynamix axis for each input / output
            training=TrainingMode.EVAL,  # always put the model in evaluation mode
            verbose=False,
        )


def convert_to_quant_onnx(
    model_pytorch: PreTrainedModel, output_path: str, inputs_pytorch: Od[str, torch.Tensor]
) -> None:
    """
    Convert a quantized Pytorch model to ONNX file.
    :param model_pytorch: Pytorch model
    :param output_path: ONNX file path
    :param inputs_pytorch: some dummy input (Pytorch tensor on the same device than the model)
    """
    from pytorch_quantization.nn import TensorQuantizer

    TensorQuantizer.use_fb_fake_quant = True
    convert_to_onnx(model_pytorch=model_pytorch, output_path=output_path, inputs_pytorch=inputs_pytorch, opset=13)
    TensorQuantizer.use_fb_fake_quant = False


def optimize_onnx(onnx_path: str, onnx_optim_model_path: str, fp16: bool, use_cuda: bool) -> None:
    """
    ONNX Runtime transformer graph optimization.
    Performs some operator fusion (merge several nodes of the graph in a single one)
    and may convert some nodes to reduced precision.
    :param onnx_path: ONNX input path
    :param onnx_optim_model_path: where to save optimized model
    :param fp16: use mixed precision (faster inference)
    :param use_cuda: perform optimization on GPU (should )
    """
    optimization_options = FusionOptions("bert")
    optimization_options.enable_gelu_approximation = False  # additional optimization
    optimized_model: BertOnnxModel = optimizer.optimize_model(
        input=onnx_path,
        model_type="bert",
        use_gpu=use_cuda,
        opt_level=1,
        num_heads=0,  # automatic detection may not work with opset 13
        hidden_size=0,  # automatic detection
        optimization_options=optimization_options,
    )
    if fp16:
        optimized_model.convert_float_to_float16()  # FP32 -> FP16
    logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(onnx_optim_model_path)


def cpu_quantization(input_model_path: str, output_model_path: str) -> None:
    """
    ONNX CPU only dynamic quantization
    :param input_model_path: ONNX graph (float) to quantize
    :param output_model_path: where to save quantized model
    """
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        op_types_to_quantize=["MatMul", "Attention"],
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        extra_options={"WeightSymmetric": False, "MatMulConstBOnly": True},
    )
