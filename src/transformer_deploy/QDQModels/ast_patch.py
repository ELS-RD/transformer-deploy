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
import code
import importlib
import inspect
import logging
from typing import List, Optional, Sequence, Tuple


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


def should_be_quantized(node: ast.AST, torch_op_to_quantize: Tuple[str]) -> bool:
    """
    Predicate to check if a torch operation should be optimized
    :param node: ast node to check
    :param torch_op_to_quantize: list of torch operations to optimize
    :return: True if node should be optimized
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr in torch_op_to_quantize
    )


def patch_nodes(head_node: ast.Module, torch_op_to_quantize: Tuple[str]) -> Tuple[ast.Module, List[str]]:
    """
    Replace an operation to optimize by its optimized version.
    May have to generate some quantization node names.
    :param head_node: ast node to modify
    :param torch_op_to_quantize: list of torch operations to optimize
    :return: the modified ast tree and the list of generated quantization nodes
    """
    q_attr_names: List[str] = list()
    for node in ast.walk(head_node):  # type: ast.Call
        if should_be_quantized(node=node, torch_op_to_quantize=torch_op_to_quantize):
            assert len(node.args) >= 2, f"unexpected number of args: {len(node.args)} args in {node.func.attr} node"
            for index in range(2):  # only apply transfo to the 2 first args
                arg = node.args[index]
                q_name = f"quantizer_{len(q_attr_names)}"
                q_attr_names.append(q_name)
                node.args[index] = wrap_attr(q_name, arg)

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


def add_quant_to_module(module_to_patch: type, new_module_name: str, torch_op_to_quantize: Tuple[str]) -> ast.Module:
    """
    Modify a class to add quantization operations around each torch operation to optimize.
    :param module_to_patch: Pytorch module to patch
    :param new_module_name: new name for the module
    :param torch_op_to_quantize: list of torch operations to optimize
    :return: modified ast tree
    """
    source_code = inspect.getsource(module_to_patch)
    head = ast.parse(source_code)
    head, nodes_to_add = patch_nodes(head, torch_op_to_quantize=torch_op_to_quantize)
    add_init_quantizer(head_node=head, q_attr_names=nodes_to_add)
    head = add_qdq_to_class_name(head_node=head, new_class_name=new_module_name)
    return head


def contains_op(node: ast.AST, torch_op_to_quantize: Tuple[str]) -> bool:
    """
    Check if a tree contains some operations to optimize.
    :param node: Head of the ast tree
    :param torch_op_to_quantize: list of Pytorch operations to optimize
    :return: True if ast tree contains operations to optimize
    """
    for node in ast.walk(node):
        if should_be_quantized(node=node, torch_op_to_quantize=torch_op_to_quantize):
            return True
    return False


def list_class_to_patch(model_module, torch_op_to_quantize: Sequence[str]) -> List[str]:
    """
    List all classes which contain operations to be optimized.
    :param model_module: Pytorch module
    :param torch_op_to_quantize: list of Pytorch operations to be optimized
    :return: the list of module names to be optimized
    """
    module_names: List[str] = list()
    module_source_code = inspect.getsource(model_module)
    head_node = ast.parse(module_source_code)
    for node in ast.walk(head_node):
        if isinstance(node, ast.ClassDef) and contains_op(node=node, torch_op_to_quantize=torch_op_to_quantize):
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
    import torch
    torch.nn.Linear = quant_nn.QuantLinear
    """
    # remove extra spaces
    import_code = inspect.cleandoc(import_code)
    # execute the code in the module context
    exec(import_code, model_module.__dict__, model_module.__dict__)


def add_quantization_to_model(
    module_path: str,
    class_to_patch: Optional[List[str]] = None,
    torch_op_to_quantize: Sequence[str] = ("matmul", "add"),
) -> None:
    """
    Add quantization support to a model.
    :param module_path: model module to optimize
    :param class_to_patch: name of modules to patch, if None it will be auto-detected.
    :param torch_op_to_quantize: list of Pytorch operations to optimize
    """
    model_module = importlib.import_module(name=module_path)
    load_missing_imports(model_module)

    if class_to_patch is None:
        class_to_patch = list_class_to_patch(model_module=model_module, torch_op_to_quantize=torch_op_to_quantize)
        logging.info(f"patch following class: {', '.join(class_to_patch)}")

    for class_name in class_to_patch:
        module_to_patch = getattr(model_module, class_name)
        head = add_quant_to_module(
            module_to_patch=module_to_patch, new_module_name=class_name, torch_op_to_quantize=torch_op_to_quantize
        )
        head = ast.fix_missing_locations(head)
        module_patched: code = compile(head, filename="<ast modif - transformer deploy>", mode="exec")
        # execute the code in the module context so it overrides the original classes and leverage existing imports
        exec(module_patched, model_module.__dict__, model_module.__dict__)
