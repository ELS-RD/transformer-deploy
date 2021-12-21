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

import importlib
from typing import Dict, List

import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor

from transformer_deploy.QDQModels.QDQAlbert import qdq_albert_mapping
from transformer_deploy.QDQModels.QDQBert import qdq_bert_mapping
from transformer_deploy.QDQModels.QDQDistilbert import qdq_distilbert_mapping
from transformer_deploy.QDQModels.QDQElectra import qdq_electra_mapping
from transformer_deploy.QDQModels.QDQRoberta import qdq_roberta_mapping
from transformer_deploy.QDQModels.utils import PatchTransformers


def patch_model(patch: PatchTransformers) -> PatchTransformers:
    backup: Dict[str, torch.nn.Module] = dict()
    model = importlib.import_module(patch.module)
    for class_name, qdq_class in patch.mapping.items():
        backup[class_name] = getattr(model, class_name)
        setattr(model, class_name, qdq_class)
    return PatchTransformers(module=patch.module, mapping=backup)


def add_qdq() -> List[PatchTransformers]:
    restore = list()
    for patch in [
        qdq_bert_mapping,
        qdq_roberta_mapping,
        qdq_electra_mapping,
        qdq_distilbert_mapping,
        qdq_albert_mapping,
    ]:
        backup = patch_model(patch)
        restore.append(backup)
    return restore


def remove_qdq(backup: List[PatchTransformers]):
    for patch in backup:
        patch_model(patch)


def setup_qat(per_channel: bool):
    axis = (0,) if per_channel else None
    input_desc = QuantDescriptor(num_bits=8, calib_method="histogram")
    # below we do per-channel quantization for weights, set axis to None to get a per tensor calibration
    weight_desc = QuantDescriptor(num_bits=8, axis=axis)
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)


# TODO
# Deberta V2 -> ? (will need to check ONNX export)
