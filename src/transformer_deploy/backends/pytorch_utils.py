#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Utils related to Pytorch inference.
"""
from collections import OrderedDict
from typing import Callable, Dict
from typing import OrderedDict as Od
from typing import Tuple

import numpy as np
import torch
from torch.onnx import TrainingMode
from transformers import AutoConfig, PreTrainedModel


def infer_classification_pytorch(
    model: PreTrainedModel, run_on_cuda: bool
) -> Callable[[Dict[str, torch.Tensor]], np.ndarray]:
    """
    Perform Pytorch inference for classification task
    :param model: Pytorch model (transformers)
    :param run_on_cuda: True if should be ran on GPU
    :return: a function to perform inference
    """

    def infer(inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        model_output = model(**inputs).logits.detach().cpu().numpy()  # noqa: F821
        if run_on_cuda:
            torch.cuda.synchronize()
        return model_output

    return infer


def infer_feature_extraction_pytorch(
    model: PreTrainedModel, run_on_cuda: bool
) -> Callable[[Dict[str, torch.Tensor]], np.ndarray]:
    """
    Perform Pytorch inference for feature extraction task
    :param model: Pytorch model (sentence-transformers)
    :param run_on_cuda: True if should be ran on GPU
    :return: a function to perform inference
    """

    def infer(inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        model_output = model(**inputs).detach().cpu().numpy()  # noqa: F821
        if run_on_cuda:
            torch.cuda.synchronize()
        return model_output

    return infer


def get_model_size(path: str) -> Tuple[int, int]:
    """
    Find number of attention heads and hidden layer size of a model
    :param path: path to model
    :return: tupple of # of attention heads and hidden layer size (0 if not found)
    """
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=path)
    num_attention_heads = getattr(config, "num_attention_heads", 0)
    hidden_size = getattr(config, "hidden_size", 0)
    return num_attention_heads, hidden_size


def convert_to_onnx(
    model_pytorch: PreTrainedModel, output_path: str, inputs_pytorch: Od[str, torch.Tensor], quantization: bool
) -> None:
    """
    Convert a Pytorch model to an ONNX graph by tracing the provided input inside the Pytorch code.
    ONNX opset 12 used for non quantized models, and 13 otherwise.
    :param model_pytorch: Pytorch model (transformers)
    :param output_path: where to save ONNX file
    :param inputs_pytorch: Tensor, can be dummy data, shape is not important as we declare all axes as dynamic.
    Should be on the same device than the model (CPU or GPU)
    :param quantization: model is quantized
    """
    if quantization:
        try:
            from pytorch_quantization.nn import TensorQuantizer
        except ImportError:
            raise ImportError(
                "It seems that pytorch-quantization is not yet installed. "
                "It is required when you enable the quantization flag and use CUDA device."
                "Please find installation instructions on "
                "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization or use:\n"
                "pip3 install git+ssh://git@github.com/NVIDIA/TensorRT#egg=pytorch-quantization\\&"
                "subdirectory=tools/pytorch-quantization/"
            )

        TensorQuantizer.use_fb_fake_quant = True

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
            opset_version=13,  # the ONNX version to use, >= 13 supports channel quantized model
            do_constant_folding=True,  # simplify model (replace constant expressions)
            input_names=list(inputs_pytorch.keys()),  # input names
            output_names=["output"],  # output axis name
            dynamic_axes=dynamic_axis,  # declare dynamix axis for each input / output
            training=TrainingMode.EVAL,  # always put the model in evaluation mode
            verbose=False,
        )
    if quantization:
        TensorQuantizer.use_fb_fake_quant = False
