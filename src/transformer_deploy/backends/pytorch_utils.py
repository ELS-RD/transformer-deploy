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

import onnx
import torch
from torch.onnx import TrainingMode
from transformers import AutoConfig, PreTrainedModel


def infer_classification_pytorch(
    model: PreTrainedModel, run_on_cuda: bool
) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """
    Perform Pytorch inference for classification task
    :param model: Pytorch model (transformers)
    :param run_on_cuda: True if should be ran on GPU
    :return: a function to perform inference
    """

    def infer(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_output = model(**inputs).logits.detach()  # noqa: F821
        if run_on_cuda:
            torch.cuda.synchronize()
        return model_output

    return infer


def infer_feature_extraction_pytorch(
    model: PreTrainedModel, run_on_cuda: bool
) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """
    Perform Pytorch inference for feature extraction task
    :param model: Pytorch model (sentence-transformers)
    :param run_on_cuda: True if should be ran on GPU
    :return: a function to perform inference
    """

    def infer(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_output = model(**inputs).detach()  # noqa: F821
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
    model_pytorch: PreTrainedModel,
    output_path: str,
    inputs_pytorch: Od[str, torch.Tensor],
    quantization: bool,
    var_output_seq: bool,
    fix_output_dim_size: bool = True,
) -> None:
    """
    Convert a Pytorch model to an ONNX graph by tracing the provided input inside the Pytorch code.
    ONNX opset 12 used for non quantized models, and 13 otherwise.
    :param model_pytorch: Pytorch model (transformers)
    :param output_path: where to save ONNX file
    :param inputs_pytorch: Tensor, can be dummy data, shape is not important as we declare all axes as dynamic.
    Should be on the same device than the model (CPU or GPU)
    :param quantization: model is quantized
    :param var_output_seq: variable size sequence
    :param fix_output_dim_size: make output axis size fixed (match pytorch model output) when its name is unknown.
        Useful to ease output tensor shape prediction.
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
    if hasattr(model_pytorch, "config") and hasattr(model_pytorch.config, "use_cache"):
        use_cache = getattr(model_pytorch.config, "use_cache")
        setattr(model_pytorch.config, "use_cache", False)

    # dynamic axis == variable length axis
    dynamic_axis = OrderedDict()
    for k in inputs_pytorch.keys():
        if var_output_seq:
            # seq axis name is fixed to be matched with output seq axis name (for output shape prediction)
            dynamic_axis[k] = {0: "batch_size", 1: "sequence"}
        else:
            # if there is no specific requirement, each axis name is unique, fix some issue on T5 model
            dynamic_axis[k] = {0: "batch_size", 1: f"sequence-{k}"}
    dynamic_axis["output"] = {0: "batch_size"}
    if var_output_seq:
        dynamic_axis["output"][1] = "sequence"
    # replace int64 input tensors by int32 -> for ONNX Runtime binding API and expected by TensorRT engine
    for k, v in inputs_pytorch.items():
        if v.dtype in [torch.long, torch.int64]:
            inputs_pytorch[k] = v.type(torch.int32)
    with torch.no_grad():
        torch.onnx.export(
            model_pytorch,  # model to optimize
            args=tuple(inputs_pytorch.values()),  # tuple of multiple inputs
            f=output_path,  # output path / file object
            opset_version=13,  # the ONNX version to use, >= 13 supports channel quantized model
            do_constant_folding=True,  # simplify model (replace constant expressions)
            input_names=list(inputs_pytorch.keys()),  # input names
            output_names=["output"],  # output axis name, hard coded so only 1 output supported
            dynamic_axes=dynamic_axis,  # declare dynamix axis for each input / output
            training=TrainingMode.EVAL,  # always put the model in evaluation mode
            verbose=False,
        )
    if quantization:
        TensorQuantizer.use_fb_fake_quant = False
    if hasattr(model_pytorch, "config") and hasattr(model_pytorch.config, "use_cache"):
        setattr(model_pytorch.config, "use_cache", use_cache)

    if fix_output_dim_size:
        # Pytorch sometimes fails to infer output tensor shape of models
        # In ONNX graph, axis name is marked like "Divoutput_dim_1" which is a generated name,
        # and there may be a warning:
        # ** "WARNING: The shape inference of prim::Constant type is missing, so it may result in wrong shape inference
        # for the exported graph. Please consider adding it in symbolic function." **
        # ex.: https://discuss.pytorch.org/t/bidirectional-lstm-and-onnx-runtime-warnings/136374
        # We need non dynamic axis to have fixed value to reserve GPU memory when using ONNX Runtime io binding API
        # Below we reopen the model and override the dynamic shape by a fixed one
        output = model_pytorch(**inputs_pytorch)
        # sentence-transformers outputs a torch tensor, Hugging Face transformers model output an ordered dictionary
        if not isinstance(output, torch.Tensor):
            output: torch.Tensor = output[0]
        assert len(output[0].shape) >= len(
            dynamic_axis["output"]
        ), f"{len(output[0].shape)} >= {len(dynamic_axis['output'])}"
        onnx_model = onnx.load(output_path)
        output_axis_name = set(dynamic_axis["output"].values())
        for index in range(len(onnx_model.graph.output[0].type.tensor_type.shape.dim)):
            if onnx_model.graph.output[0].type.tensor_type.shape.dim[index].dim_param in output_axis_name:
                continue
            onnx_model.graph.output[0].type.tensor_type.shape.dim[index].dim_value = output.shape[index]
        onnx.save(onnx_model, output_path)
