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

import ast
import importlib
import logging

import torch
from torch import nn

from transformer_deploy.QDQModels.ast_module_patch import add_quant_to_module, list_class_to_patch
from transformer_deploy.QDQModels.ast_operator_patch import Patch2ArgsNode, PatchAdd2ArgsNode


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=5, out_features=5, bias=True)

    def forward(self, inputs: torch.Tensor):
        a: torch.Tensor = self.linear(inputs)
        b = torch.ones(a.shape)
        c = torch.matmul(a, b)
        d = nn.LayerNorm(a + c)
        return d


expected_class = """
class QDQFakeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=5, out_features=5, bias=True)
        self.quantizer_0 = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.quantizer_1 = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.quantizer_2 = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.quantizer_3 = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)

    def forward(self, inputs: torch.Tensor):
        a: torch.Tensor = self.linear(inputs)
        b = torch.ones(a.shape)
        c = torch.matmul(self.quantizer_0(a), self.quantizer_1(b))
        d = nn.LayerNorm(self.quantizer_2(a) + self.quantizer_3(c))
        return d
""".strip()


def test_list_class():
    model_module = importlib.import_module(name=__name__)
    class_to_patch = list_class_to_patch(model_module=model_module)
    assert class_to_patch == ["FakeModel"]


def test_add_quant():
    head = add_quant_to_module(module_to_patch=FakeModel, new_module_name="QDQFakeModel")
    head = ast.fix_missing_locations(head)
    logging.error(ast.unparse(head))
    assert ast.unparse(head) == expected_class


def test_patch_2_args_node():
    source_code = "torch.matmul(a, b)"
    patch = Patch2ArgsNode(op="matmul")
    head: ast.AST = ast.parse(source_code).body[0].value
    assert patch.should_patch(head)
    head_patched = patch.patch(node=head, nb_quant_node=0)
    assert ast.unparse(head) == "torch.matmul(self.quantizer_0(a), self.quantizer_1(b))"
    assert head_patched == ["quantizer_0", "quantizer_1"]


def test_add_2_args_node():
    source_code = "nn.LayerNorm(hidden_states + input_tensor)"
    patch = PatchAdd2ArgsNode(op="LayerNorm")
    head: ast.AST = ast.parse(source_code).body[0].value
    assert patch.should_patch(head)
    head_patched = patch.patch(node=head, nb_quant_node=0)
    assert ast.unparse(head) == "nn.LayerNorm(self.quantizer_0(hidden_states) + self.quantizer_1(input_tensor))"
    assert head_patched == ["quantizer_0", "quantizer_1"]
