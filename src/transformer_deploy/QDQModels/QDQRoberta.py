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
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
"""
This module add quantization support to all Roberta architecture based models.
"""


import torch
import torch.utils.checkpoint

from transformer_deploy.QDQModels.ast_utils import PatchModule


def qdq_create_position_tensorrt(input_ids, padding_idx, past_key_values_length=0):
    """
    Override qdq_create_position_tensorrt function.
    It appeared that cumsum operator in TensorRT doesn't support integer type.
    see https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md
    This override uses float instead.
    """
    # QDQ change below
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    # int() -> float() because of a limitations in cumsum operator implementation in TensorRT
    mask = input_ids.ne(padding_idx).float()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


qdq_roberta_mapping: PatchModule = PatchModule(
    module="transformers.models.roberta.modeling_roberta",
    monkey_patch={
        "create_position_ids_from_input_ids": (qdq_create_position_tensorrt, "qdq_create_position_tensorrt"),
    },
)
