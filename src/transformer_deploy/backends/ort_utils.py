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
from typing import List, Union

from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel


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
