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

import inspect
from dataclasses import dataclass
from typing import Dict, List

import torch
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from tqdm import trange
from transformers import PreTrainedModel


@dataclass
class PatchTransformers:
    module: str
    mapping: Dict[str, torch.nn.Module]

    def print_code(self):
        for class_name, cl in self.mapping.items():
            print("---------")
            print(class_name)
            inspect.getsource(cl)


def setup_nvidia_qat(per_channel_q: bool = True, method: str = "histogram"):
    assert method in ["histogram", "max"], f"unknown calibration method (for NLP): {method}"
    axis = (0,) if per_channel_q else None
    input_desc = QuantDescriptor(num_bits=8, calib_method=method)
    weight_desc = QuantDescriptor(num_bits=8, axis=axis)
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)


def calibrate_qdq_nodes(
    model: PreTrainedModel, encoded_dataset: List[Dict[str, torch.Tensor]], batch_size: int, nb_sample: int = 128
) -> PreTrainedModel:
    model = model.cpu()
    # Find the TensorQuantizer and enable calibration
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    with torch.no_grad():
        for start_index in trange(0, nb_sample, batch_size):
            end_index = start_index + batch_size
            data = encoded_dataset[start_index:end_index]
            for d in data:
                input_torch = {k: torch.tensor(v, dtype=torch.long, device="cpu") for k, v in d.items()}
                model(**input_torch)

    # Finalize calibration
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    # strict=False -> avoid Exception when some quantizer are never used
                    # (because of a condition for instance)
                    module.load_calib_amax("percentile", percentile=99.99, strict=False)
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

    model.cuda()
    return model
