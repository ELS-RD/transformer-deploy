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
Simple to use wrapper to patch transformer models AST
"""

import importlib
import inspect
import logging
from typing import List, Optional

from transformer_deploy.QDQModels.ast_utils import PatchModule, add_quantization_to_model
from transformer_deploy.QDQModels.QDQAlbert import qdq_albert_mapping
from transformer_deploy.QDQModels.QDQBert import qdq_bert_mapping
from transformer_deploy.QDQModels.QDQDeberta import qdq_deberta_mapping, qdq_deberta_v2_mapping
from transformer_deploy.QDQModels.QDQDistilbert import qdq_distilbert_mapping
from transformer_deploy.QDQModels.QDQElectra import qdq_electra_mapping
from transformer_deploy.QDQModels.QDQRoberta import qdq_roberta_mapping


tested_models: List[PatchModule] = [
    qdq_bert_mapping,
    qdq_roberta_mapping,
    qdq_electra_mapping,
    qdq_distilbert_mapping,
    qdq_albert_mapping,
    qdq_deberta_mapping,  # quantization is ok, ONNX export doesn't work
    qdq_deberta_v2_mapping,  # quantization is ok, ONNX export doesn't work
]


def patch_model(patch: PatchModule) -> None:
    """
    Perform modifications to model to make it work with ONNX export and quantization.
    :param patch: an object containing all the information to perform a modification
    """
    add_quantization_to_model(module_path=patch.module, class_to_patch=None)
    model_module = importlib.import_module(patch.module)
    for target, (modified_object, object_name) in patch.monkey_patch.items():
        source_code = inspect.getsource(modified_object)
        source_code += f"\n{target} = {object_name}"
        exec(source_code, model_module.__dict__, model_module.__dict__)


def add_qdq(modules_to_patch: Optional[List[PatchModule]] = None) -> None:
    """
    Add quantization support to each tested model by modifyin their AST.
    :param modules_to_patch: list of operator to target
    """
    if modules_to_patch is None:
        modules_to_patch = tested_models
    for patch in modules_to_patch:
        logging.info(f"add quantization to module {patch.module}")
        patch_model(patch)


def remove_qdq(modules_to_patch: Optional[List[PatchModule]] = None) -> None:
    """
    Restore AST of modified modules.
    :param modules_to_patch: list of operator to target
    """
    if modules_to_patch is None:
        modules_to_patch = tested_models
    for patch in modules_to_patch:
        logging.info(f"restore module {patch.module}")
        patch.restore()
