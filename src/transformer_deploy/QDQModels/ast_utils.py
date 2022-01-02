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
Contains the code to patch model AST in RAM.
"""

import ast
import code
import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from transformer_deploy.QDQModels.ast_operator_patch import Patch2ArgsNode, PatchAdd2ArgsNode, PatchLayer, PatchNode


# list of Pytorch operations to optimize, you can reduce it to increase PTQ/QAT accuracy
op_to_quant: List[PatchNode] = [
    Patch2ArgsNode(op="matmul"),
    Patch2ArgsNode(op="add"),
    Patch2ArgsNode(op="bmm"),
    PatchAdd2ArgsNode(op="LayerNorm"),
    PatchLayer(origin_module="nn", origin_layer="Linear", target_module="quant_nn", target_layer="QuantLinear"),
]


@dataclass
class PatchModule:
    module: str
    monkey_patch: Dict[str, Tuple[Callable, str]] = field(default_factory=dict)

    def print_code(self):
        for class_name, cl in self.monkey_patch.items():
            print("---------")
            print(class_name)
            inspect.getsource(cl)

    def restore(self):
        model_module = importlib.import_module(name=self.module)
        importlib.reload(model_module)


def init_quantizer(name: str) -> ast.Assign:
    """
    Generate quantization node initialization to add to the end of __init__()
    :param name: generated name of the node
    :return: quantization init ast node
    """
    quant_linear = ast.Attribute(value=ast.Name(id="quant_nn", ctx=ast.Load()), attr="QuantLinear", ctx=ast.Load())
    default_quant_desc_input = ast.Attribute(value=quant_linear, attr="default_quant_desc_input", ctx=ast.Load())
    tensor_quant = ast.Name(id="TensorQuantizer", ctx=ast.Load())
    quant_value = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=name, ctx=ast.Store())
    return ast.Assign(
        targets=[quant_value],
        value=ast.Call(func=tensor_quant, args=[default_quant_desc_input], keywords=[]),
    )


def patch_nodes(head_node: ast.Module) -> Tuple[ast.Module, List[str]]:
    """
    Replace an operation to optimize by its optimized version.
    May have to generate some quantization node names.
    :param head_node: ast node to modify
    :return: the modified ast tree and the list of generated quantization nodes
    """
    q_attr_names: List[str] = list()
    for node in ast.walk(head_node):  # type: ast.Call
        for op in op_to_quant:
            if op.should_patch(node=node):
                quant_names = op.patch(node=node, nb_quant_node=len(q_attr_names))
                q_attr_names.extend(quant_names)

    return head_node, q_attr_names


def add_init_quantizer(head_node: ast.Module, q_attr_names: List[str]) -> ast.Module:
    """
    Add initialization of quantizer to __init__()
    :param head_node: node related to a class to optimize
    :param q_attr_names: list of quantizer names to init
    :return: modified ast tree
    """
    for node in ast.walk(head_node):  # type: ast.FunctionDef
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for name in q_attr_names:
                quantizer = init_quantizer(name)
                node.body.append(quantizer)
    return head_node


def add_qdq_to_class_name(head_node: ast.Module, new_class_name: str) -> ast.Module:
    """
    Change the name of the class to optimize (may help in debugging / error messages)
    :param head_node: node related to the class to optimize
    :param new_class_name: new name to use
    :return: the modified ast tree
    """
    for node in ast.walk(head_node):  # type: ast.ClassDef
        if isinstance(node, ast.ClassDef):
            node.name = new_class_name
    return head_node


def add_quant_to_module(module_to_patch: type, new_module_name: str) -> ast.Module:
    """
    Modify a class to add quantization operations around each torch operation to optimize.
    :param module_to_patch: Pytorch module to patch
    :param new_module_name: new name for the module
    :return: modified ast tree
    """
    source_code = inspect.getsource(module_to_patch)
    head = ast.parse(source_code)
    head, nodes_to_add = patch_nodes(head)
    add_init_quantizer(head_node=head, q_attr_names=nodes_to_add)
    head = add_qdq_to_class_name(head_node=head, new_class_name=new_module_name)
    return head


def contains_op(node: ast.AST) -> bool:
    """
    Check if a tree contains some operations to optimize.
    :param node: Head of the ast tree
    :return: True if ast tree contains operations to optimize
    """
    for node in ast.walk(node):
        for op in op_to_quant:
            if op.should_patch(node=node):
                return True
    return False


def list_class_to_patch(model_module) -> List[str]:
    """
    List all classes which contain operations to be optimized.
    :param model_module: Pytorch module
    :return: the list of module names to be optimized
    """
    module_names: List[str] = list()
    module_source_code = inspect.getsource(model_module)
    head_node = ast.parse(module_source_code)
    for node in ast.walk(head_node):
        if isinstance(node, ast.ClassDef) and contains_op(node=node):
            module_names.append(node.name)
    return module_names


def load_missing_imports(model_module) -> None:
    """
    Execute some imports in the context of a module.
    Override Linear layer by its quantized version
    :param model_module: module to use for the imports
    """
    import_code = """
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.nn import TensorQuantizer
    """
    # remove extra spaces
    import_code = inspect.cleandoc(import_code)
    # execute the code in the module context
    exec(import_code, model_module.__dict__, model_module.__dict__)


def add_quantization_to_model(
    module_path: str,
    class_to_patch: Optional[List[str]],
):
    """
    Add quantization support to a model.
    :param module_path: model module to optimize
    :param class_to_patch: name of modules to patch, if None it will be auto-detected.
    :return: backup of original classes
    """
    model_module = importlib.import_module(name=module_path)
    load_missing_imports(model_module)

    if class_to_patch is None or len(class_to_patch) == 0:
        class_to_patch = list_class_to_patch(model_module=model_module)
        logging.info(f"modify class {', '.join(class_to_patch)}")

    for class_name in class_to_patch:
        module_to_patch = getattr(model_module, class_name)
        head = add_quant_to_module(module_to_patch=module_to_patch, new_module_name=class_name)
        head = ast.fix_missing_locations(head)
        module_patched: code = compile(head, filename="<ast modif - transformer deploy>", mode="exec")
        # execute the code in the module context so it overrides the original classes and leverage existing imports
        exec(module_patched, model_module.__dict__, model_module.__dict__)
