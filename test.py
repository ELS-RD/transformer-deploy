# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging as log
import time
from inspect import signature
from itertools import chain
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import onnxruntime
import torch
from onnxruntime import InferenceSession, IOBinding, OrtValue, SessionOptions
from packaging.version import parse
from torch.onnx import export
from transformers import AutoTokenizer, GPT2LMHeadModel, PreTrainedModel, TensorType, TFPreTrainedModel
from transformers.models.gpt2 import GPT2OnnxConfig
from transformers.onnx.features import FeaturesManager
from transformers.utils import logging

from transformer_deploy.backends.ort_utils import optimize_onnx


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
log.basicConfig()
log.getLogger().setLevel(logging.INFO)

# This is the minimal required version to support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = parse("1.4.0")


def ensure_model_and_config_inputs_match(
    model: Union[PreTrainedModel, TFPreTrainedModel], model_inputs: Iterable[str]
) -> Tuple[bool, List[str]]:
    """

    :param model_inputs:
    :param config_inputs:
    :return:
    """
    forward_parameters = signature(model.forward).parameters
    model_inputs_set = set(model_inputs)

    # We are fine if config_inputs has more keys than model_inputs
    forward_inputs_set = set(forward_parameters.keys())
    is_ok = model_inputs_set.issubset(forward_inputs_set)

    # Make sure the input order match (VERY IMPORTANT !!!!)
    matching_inputs = forward_inputs_set.intersection(model_inputs_set)
    ordered_inputs = [parameter for parameter in forward_parameters.keys() if parameter in matching_inputs]
    return is_ok, ordered_inputs


model_name = "gpt2"
feature = "causal-lm"
# feature = "causal-lm-with-past"
seq_len = 256
atol = 0.2

# causal-lm-with-past, 256 tokens, no opt: 0.01640296936035156 , opt: 0.07701282262802124
# (no atol error but opt strange messages)
# with cache: no opt: 0.41653642654418943
# causal-lm, 256 tokens, no opt: 0.00821619749069214, opt: 0.005052394866943359 (atol error 0.3, gap 0.38)
# opt (CUDA) FP32: 0.00902557373046875 (no atol error  0.3, gap 0.11)
# with CPU: no opt: 0.5370134830474853, opt:

tokenizer = AutoTokenizer.from_pretrained(model_name)
model: GPT2LMHeadModel = FeaturesManager.get_model_from_feature(feature, model_name)
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config: GPT2OnnxConfig = model_onnx_config(model.config)

with torch.no_grad():
    model.config.return_dict = True
    model.eval()

    # Check if we need to override certain configuration item
    if onnx_config.values_override is not None:
        logger.info(f"Overriding {len(onnx_config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in onnx_config.values_override.items():
            logger.info(f"\t- {override_config_key} -> {override_config_value}")
            setattr(model.config, override_config_key, override_config_value)

    # Ensure inputs match
    model_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH)
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.int64:
            model_inputs[k] = model_inputs[k].type(torch.int32)
    inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
    onnx_outputs = list(onnx_config.outputs.keys())

    if not inputs_match:
        raise ValueError("Model and config inputs doesn't match")

    onnx_config.patch_ops()

    # export can works with named args but the dict containing named args as to be last element of the args tuple
    export(
        model,
        (model_inputs,),
        f="test-export.onnx",
        input_names=list(onnx_config.inputs.keys()),
        output_names=onnx_outputs,
        dynamic_axes={name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())},
        do_constant_folding=True,
        use_external_data_format=onnx_config.use_external_data_format(model.num_parameters()),
        enable_onnx_checker=True,
        opset_version=13,
    )

    onnx_config.restore_ops()

onnx_inputs, onnx_named_outputs = matched_inputs, onnx_outputs


# validate_model_outputs(onnx_config, tokenizer, model, Path("test-export.onnx"), onnx_named_outputs, 0.2)
optimize_onnx(
    onnx_path="test-export.onnx",
    onnx_optim_model_path="test-export-opt.onnx",
    fp16=True,
    use_cuda=True,
    num_attention_heads=onnx_config.num_attention_heads,
    hidden_size=model.config.n_embd,
    architecture="gpt2",
)

reference_model_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH, seq_length=seq_len)

# Create ONNX Runtime session
options = SessionOptions()
session = InferenceSession(path_or_bytes="test-export.onnx", sess_options=options, providers=["CUDAExecutionProvider"])

# Compute outputs from the reference model
ref_outputs = model(**reference_model_inputs)
ref_outputs_dict = {}

# We flatten potential collection of outputs (i.e. past_keys) to a flat structure
for name, value in ref_outputs.items():
    # Overwriting the output name as "present" since it is the name used for the ONNX outputs
    # ("past_key_values" being taken for the ONNX inputs)
    if name == "past_key_values":
        name = "present"
    if isinstance(value, (list, tuple)):
        value = onnx_config.flatten_output_collection_property(name, value)
        ref_outputs_dict.update(value)
    else:
        ref_outputs_dict[name] = value

# We flatten potential collection of inputs (i.e. past_keys)
onnx_inputs = {}
for name, value in reference_model_inputs.items():
    if isinstance(value, (list, tuple)):
        value = onnx_config.flatten_output_collection_property(name, value)
        onnx_inputs.update({tensor_name: pt_tensor.numpy() for tensor_name, pt_tensor in value.items()})
    else:
        onnx_inputs[name] = value.numpy()

for k, v in onnx_inputs.items():  # type: str, np.ndarray
    if isinstance(v, np.ndarray) and v.dtype == np.int64:
        onnx_inputs[k] = v.astype(np.int32)
    print(f"{k}: {type(v)} -> {v.shape} # {v.dtype}")

# Compute outputs from the ONNX model
onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

# Check we have a subset of the keys into onnx_outputs against ref_outputs
ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
if not onnx_outputs_set.issubset(ref_outputs_set):
    raise ValueError(
        "Outputs doesn't match between reference model and ONNX exported model: "
        f"{onnx_outputs_set.difference(ref_outputs_set)}"
    )
else:
    logger.info(f"\t-[✓] ONNX model outputs' name match reference model ({onnx_outputs_set})")

# Check the shape and values match
for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
    ref_value = ref_outputs_dict[name].detach().numpy()
    # Shape
    if not ort_value.shape == ref_value.shape:
        raise ValueError(
            "Outputs shape doesn't match between reference model and ONNX exported model: "
            f"Got {ref_value.shape} (reference) and {ort_value.shape} (ONNX)"
        )
    else:
        logger.info(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

    # Values
    if not np.allclose(ref_value, ort_value, atol=atol):
        raise ValueError(
            "Outputs values doesn't match between reference model and ONNX exported model: "
            f"Got max absolute difference of: {np.amax(np.abs(ref_value - ort_value))}"
        )
    else:
        logger.info(f"\t\t-[✓] all values close (atol: {atol})")


# test export to Torch tensor directly
# https://onnxruntime.ai/docs/api/python/api_summary.html#iobinding


def inference_onnx_binding(inputs: Dict[str, np.ndarray], session: InferenceSession, device_name: str) -> torch.Tensor:
    binding: IOBinding = session.io_binding()
    for name, tensor in inputs.items():  # type: str, np.ndarray
        if tensor.dtype in [np.int64, int]:
            tensor = tensor.astype(np.int32)
        ort_value: OrtValue = onnxruntime.OrtValue.ortvalue_from_numpy(tensor, device_name, 0)
        binding.bind_input(name, device_name, 0, tensor.dtype, ort_value.shape(), ort_value.data_ptr())
    batch_size, nb_tokens = inputs["input_ids"].shape
    output = torch.empty((batch_size, nb_tokens, tokenizer.vocab_size), dtype=torch.float32, device=device_name)
    binding.bind_output(
        name=onnx_named_outputs[0],
        device_type=device_name,
        device_id=0,
        element_type=np.float32,
        shape=tuple(output.shape),
        buffer_ptr=output.data_ptr(),
    )

    session.run_with_iobinding(binding)
    return output


inputs = tokenizer(
    "Here is some text to encode Hello World " * 30,
    add_special_tokens=True,
    return_attention_mask=True,
    return_tensors="np",
)


start = time.time()
nb_xp = 100
device_name = "cuda"
for _ in range(nb_xp):
    inference_onnx_binding(inputs=inputs, session=session, device_name=device_name)
print((time.time() - start) / nb_xp)
