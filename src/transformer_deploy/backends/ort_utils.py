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

import logging
import multiprocessing
from collections import OrderedDict
from typing import OrderedDict as OD

import torch
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from torch.onnx import TrainingMode
from transformers import PreTrainedModel


def create_model_for_provider(path: str, provider_to_use: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if type(provider_to_use) == list and provider_to_use == "CPUExecutionProvider":
        options.intra_op_num_threads = multiprocessing.cpu_count()
    if type(provider_to_use) != list:
        provider_to_use = [provider_to_use]
    return InferenceSession(path, options, providers=provider_to_use)


def convert_to_onnx(
    model_pytorch: PreTrainedModel, output_path: str, inputs_pytorch: OD[str, torch.Tensor], opset: int = 12
) -> None:
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


def optimize_onnx(onnx_path: str, onnx_optim_fp16_path: str, use_cuda: bool) -> None:
    optimization_options = FusionOptions("bert")
    optimization_options.enable_gelu_approximation = True  # additional optimization
    optimized_model: BertOnnxModel = optimizer.optimize_model(
        input=onnx_path,
        model_type="bert",
        use_gpu=use_cuda,
        opt_level=1,
        num_heads=0,  # automatic detection don't work with opset 13
        hidden_size=0,  # automatic detection
        optimization_options=optimization_options,
    )

    optimized_model.convert_float_to_float16()  # FP32 -> FP16
    logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(onnx_optim_fp16_path)
