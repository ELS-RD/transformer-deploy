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
import inspect
from dataclasses import dataclass
from typing import Callable, Dict

import torch

from transformer_deploy.QDQModels.QDQBert import (
    QDQBertIntermediate,
    QDQBertOutput,
    QDQBertSelfAttention,
    QDQBertSelfOutput,
)
from transformer_deploy.QDQModels.QDQRoberta import (
    QDQRobertaIntermediate,
    QDQRobertaOutput,
    QDQRobertaSelfAttention,
    QDQRobertaSelfOutput,
    qdq_create_position_tensorrt,
)


@dataclass
class PatchBackup:
    module: str
    mapping: Dict[str, torch.nn.Module]


def patch_model(module: str, qdq_classes: Dict[str, torch.nn.Module]) -> PatchBackup:
    backup: Dict[str, torch.nn.Module] = dict()
    model = importlib.import_module(module)
    for class_name, qdq_class in qdq_classes.items():
        backup[class_name] = getattr(model, class_name)
        setattr(model, class_name, qdq_class)
    return PatchBackup(module=module, mapping=backup)


def unpatch_model(backup: PatchBackup):
    model = importlib.import_module(backup.module)
    for class_name, original_class in backup.mapping.items():
        setattr(model, class_name, original_class)


def patch_roberta() -> Dict[str, torch.nn.Module]:
    qdq_classes: Dict[str, torch.nn.Module] = {
        "RobertaSelfAttention": QDQRobertaSelfAttention,
        "RobertaSelfOutput": QDQRobertaSelfOutput,
        "RobertaIntermediate": QDQRobertaIntermediate,
        "RobertaOutput": QDQRobertaOutput,
        "create_position_ids_from_input_ids": qdq_create_position_tensorrt,
    }
    return patch_model(module="transformers.models.roberta.modeling_roberta", qdq_classes=qdq_classes)


def patch_bert() -> Dict[str, torch.nn.Module]:
    qdq_classes: Dict[str, torch.nn.Module] = {
        "BertSelfAttention": QDQBertSelfAttention,
        "BertSelfOutput": QDQBertSelfOutput,
        "BertIntermediate": QDQBertIntermediate,
        "BertOutput": QDQBertOutput,
    }
    return patch_model(module="transformers.models.bert.modeling_bert", qdq_classes=qdq_classes)


def get_function_content(c: type) -> str:
    return inspect.getsource(c)


def patch_transformers():
    for patch_fun in [patch_bert, patch_roberta]:  # type: Callable
        patch_fun()


# TODO
# Electra -> easy
# DistillBert -> ?
# Deberta V2 -> ? (will need to check ONNX export)
# albert -> ?
