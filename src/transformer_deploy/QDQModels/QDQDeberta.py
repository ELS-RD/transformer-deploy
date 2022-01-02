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
This module add quantization support to all Deberta architecture based models.
For now, Deberta export to ONNX doesn't work well.
This PR may help: https://github.com/microsoft/DeBERTa/pull/6
"""

import torch

from transformer_deploy.QDQModels.ast_utils import PatchModule


def get_attention_mask(self, attention_mask):
    """
    Override existing get_attention_mask method in DebertaV2Encoder class.
    This one uses signed integers instead of unsigned one.
    """
    if attention_mask.dim() <= 2:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        # unecessary conversion, byte == unsigned integer -> not supported by TensorRT
        # attention_mask = attention_mask.byte()
    elif attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1)

    return attention_mask


def symbolic(g, self, mask, dim):
    """
    Override existing symbolic static function of Xsoftmax class.
    This one uses signed integers instead of unsigned one.
    Symbolic function are used during ONNX conversion instead of Pytorch code.
    """
    import torch.onnx.symbolic_helper as sym_help
    from torch.onnx.symbolic_opset9 import masked_fill, softmax

    mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
    # r_mask = g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value)
    # replace Byte by Char to get signed numbers
    r_mask = g.op(
        "Cast",
        g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
        to_i=sym_help.cast_pytorch_to_onnx["Char"],
    )
    output = masked_fill(g, self, r_mask, g.op("Constant", value_t=torch.tensor(float("-inf"))))
    output = softmax(g, output, dim)
    return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8)))


qdq_deberta_mapping: PatchModule = PatchModule(
    module="transformers.models.deberta.modeling_deberta",
    monkey_patch={
        "XSoftmax.symbolic": (symbolic, "symbolic"),
        "DebertaEncoder.get_attention_mask": (get_attention_mask, "get_attention_mask"),
    },
)


qdq_deberta_v2_mapping: PatchModule = PatchModule(
    module="transformers.models.deberta_v2.modeling_deberta_v2",
    monkey_patch={
        "XSoftmax.symbolic": (symbolic, "symbolic"),
        "DebertaV2Encoder.get_attention_mask": (get_attention_mask, "get_attention_mask"),
    },
)
