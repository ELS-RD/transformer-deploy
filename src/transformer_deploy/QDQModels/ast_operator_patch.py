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

import abc
import ast
from typing import List


class PatchNode(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def should_patch(self, node: ast.AST) -> bool:
        """Check if a node should be patched"""
        raise Exception("to implement")

    @abc.abstractmethod
    def patch(self, node: ast.AST, count: int) -> List[str]:
        """Patch ndoe"""
        raise Exception("to implement")

    @staticmethod
    def wrap_attr(quantizer_name: str, tensor_var: ast.expr) -> ast.Call:
        """
        Generate quantization wrapping each attribute of a torch operation to optimize (matmul, add, etc.)
        :param quantizer_name: generated quantization name
        :param tensor_var: the variable to wrap
        :return: the ast tree to replace the original variable
        """
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=quantizer_name, ctx=ast.Load()),
            args=[tensor_var],
            keywords=[],
        )

    @staticmethod
    def get_quant_name(node_id: int) -> str:
        return f"quantizer_{node_id}"


class Patch2ArgsNode(PatchNode):
    def __init__(self, op: str):
        """
        Patch source code in the form torch.op(a, b) to torch.op(self.q1(a), self.q1(b))
        :param op: operator to match
        """
        self.torch_op_to_quantize = op

    def should_patch(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == self.torch_op_to_quantize
        )

    def patch(self, node: ast.AST, nb_quant_node: int) -> List[str]:
        q_attr_names = list()
        for index in range(2):  # only apply transfo to the 2 first args
            arg = node.args[index]
            q_name = self.get_quant_name(nb_quant_node + len(q_attr_names))
            q_attr_names.append(q_name)
            node.args[index] = self.wrap_attr(q_name, arg)
        return q_attr_names


class PatchAdd2ArgsNode(PatchNode):
    def __init__(self, op: str):
        """
        Patch source code in the form torch.op(a + b) to torch.op(self.q1(a) + self.q1(b))
        :param op: operator to match
        """
        self.torch_op_to_quantize = op

    def should_patch(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == self.torch_op_to_quantize
            and isinstance(node.args, list)
            and len(node.args) == 1
            and isinstance(node.args[0], ast.BinOp)
            and isinstance(node.args[0].op, ast.Add)
        )

    def patch(self, node: ast.AST, nb_quant_node: int) -> List[str]:
        left_name = self.get_quant_name(nb_quant_node)
        right_name = self.get_quant_name(nb_quant_node + 1)
        node.args[0].left = self.wrap_attr(left_name, node.args[0].left)
        node.args[0].right = self.wrap_attr(right_name, node.args[0].right)
        return [left_name, right_name]
