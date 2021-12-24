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
import logging
from typing import List

from transformer_deploy.QDQModels.ast_utils import PatchTransformers, add_quantization_to_model
from transformer_deploy.QDQModels.QDQAlbert import qdq_albert_mapping
from transformer_deploy.QDQModels.QDQBert import qdq_bert_mapping
from transformer_deploy.QDQModels.QDQDeberta import qdq_deberta_mapping
from transformer_deploy.QDQModels.QDQDistilbert import qdq_distilbert_mapping
from transformer_deploy.QDQModels.QDQElectra import qdq_electra_mapping
from transformer_deploy.QDQModels.QDQRoberta import qdq_roberta_mapping


def patch_model(patch: PatchTransformers) -> PatchTransformers:
    """
    Perform modification to model to make it work with ONNX export and quantization.
    :param patch: an object containing all the information to perform a modification
    :return: all information to restore state before modification
    """
    backup = add_quantization_to_model(
        module_path=patch.module, class_to_patch=None, torch_op_to_quantize=("matmul", "add", "bmm")
    )
    model_module = importlib.import_module(patch.module)
    for target, (modified_object, object_name) in patch.monkey_patch.items():
        source_code = inspect.getsource(modified_object)
        source_code += f"\n{target} = {object_name}"
        exec(source_code, model_module.__dict__, model_module.__dict__)
    return backup


def add_qdq() -> List[PatchTransformers]:
    restore = list()
    for patch in [
        qdq_bert_mapping,
        qdq_roberta_mapping,
        qdq_electra_mapping,
        qdq_distilbert_mapping,
        qdq_albert_mapping,
        qdq_deberta_mapping,
    ]:
        logging.info(f"add quantization to module {patch.module}")
        backup = patch_model(patch)
        restore.append(backup)
    return restore


def remove_qdq(backup: List[PatchTransformers]) -> None:
    for patch in backup:
        patch_model(patch)
