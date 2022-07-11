#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxoptimizer
from onnx import ModelProto
from onnx.shape_inference import infer_shapes_path


def save_onnx(proto: onnx.ModelProto, model_path: str, clean: bool = True) -> None:
    """
    Save Onnx file with external data support when required.

    :param proto: Onnx model
    :param clean: clean the model before saving
    :param model_path: output path
    """
    # protobuff doesn't support files > 2Gb, in this case, weights are stored in another binary file
    if clean:
        to_save = clean_graph(proto=proto)
    else:
        to_save = proto
    save_external_data: bool = to_save.ByteSize() > 2 * 1024**3
    filename = Path(model_path).name
    onnx.save_model(
        proto=to_save,
        f=model_path,
        save_as_external_data=save_external_data,
        all_tensors_to_one_file=True,
        location=filename + ".data",
    )
    infer_shapes_path(model_path=model_path, output_path=model_path)


def clean_graph(proto: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove unused nodes and unused initializers.
    May help TensorRT when it refuses to load a model.

    :param proto: Onnx model
    :return: Onnx model
    """
    # operations that are tested with transformers models
    all_optimizations = [
        "eliminate_deadend",
        "eliminate_duplicate_initializer",
        "eliminate_identity",
        "eliminate_nop_cast",
        "eliminate_nop_dropout",
        "eliminate_nop_flatten",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
    ]

    cleaned_model: onnx.ModelProto = onnxoptimizer.optimize(model=proto, passes=all_optimizations)
    return cleaned_model


def merge_autoregressive_model_graphs(model_cache_path: str, model_no_cache_path: str, output_path: str) -> None:
    """
    Merge 2 Onnx models together through If and serialize them.

    :param model_cache_path: model in the Then branch
    :param model_no_cache_path: model in the Else branch
    :param output_path: path where to serialize the model
    """
    model_cache = onnx.load_model(f=model_cache_path, load_external_data=True)
    model_no_cache = onnx.load_model(f=model_no_cache_path, load_external_data=True)

    prefix_cache = "cache_node_"
    mapping_initializer_cache_to_no_cache = dict()

    # search for not-duplicated weights, called initializer in ONNX
    to_add = list()
    # speed-up the duplicated weights search by using a dict of weights hashes
    initializer_no_cache = defaultdict(list)
    for node_no_cache in model_no_cache.graph.initializer:
        if len(node_no_cache.raw_data) < 1024**2:  # skip weights smaller than 1MB
            continue
        initializer_no_cache[hash(node_no_cache.raw_data)].append(node_no_cache)

    for initializer_cache in model_cache.graph.initializer:
        is_initializer_shared_with_no_cache_model = False
        for node_no_cache in initializer_no_cache[hash(initializer_cache.raw_data)]:
            if initializer_cache.raw_data == node_no_cache.raw_data:
                is_initializer_shared_with_no_cache_model = True
                mapping_initializer_cache_to_no_cache[initializer_cache.name] = node_no_cache.name
                break
        if not is_initializer_shared_with_no_cache_model:
            initializer_cache.name = prefix_cache + initializer_cache.name
            to_add.append(initializer_cache)
            mapping_initializer_cache_to_no_cache[initializer_cache.name] = initializer_cache.name

    model_no_cache.graph.initializer.extend(to_add)
    # I/O model names should not be prefixed
    model_io_names = [n.name for n in list(model_cache.graph.input) + list(model_cache.graph.output)]

    # replace pointers to duplicated weights to their deduplicated version
    for node in model_cache.graph.node:
        if "Identity_1" == node.name:
            print("")

        for index, input_name in enumerate(node.input):
            if input_name in model_io_names:  # check if node input is an input of the model
                continue
            node.input[index] = mapping_initializer_cache_to_no_cache.get(input_name, prefix_cache + input_name)
        for index, output_name in enumerate(node.output):
            if output_name in model_io_names:  # check if node output is an output of the model
                continue
            node.output[index] = prefix_cache + output_name
        node.name = prefix_cache + node.name
    model_io_names = [n.name for n in list(model_cache.graph.input) + list(model_cache.graph.output)]

    # prefix nodes to avoid naming collision
    prefix_cache = "init_"
    cache = dict()
    for node in model_no_cache.graph.initializer:
        if node.name in model_io_names:
            new_name = prefix_cache + node.name
            cache[node.name] = new_name
            node.name = new_name

    for node in model_no_cache.graph.node:
        for input_index, n in enumerate(node.input):
            node.input[input_index] = cache.get(n, n)

    # mandatory for subgraph in if/else node
    assert len(model_cache.graph.output) == len(
        model_no_cache.graph.output
    ), f"{len(model_cache.graph.output)} vs {len(model_no_cache.graph.output)}"

    # build a computation graph with cache support
    graph_cache: onnx.GraphProto = onnx.helper.make_graph(
        nodes=list(model_cache.graph.node),
        name="graph-cache",
        inputs=[],
        outputs=list(model_cache.graph.output),
        initializer=[],
    )

    # build a computation which doesn't need past states to run
    graph_no_cache: onnx.GraphProto = onnx.helper.make_graph(
        nodes=list(model_no_cache.graph.node),
        name="graph-no-cache",
        inputs=[],
        outputs=list(model_no_cache.graph.output),
        initializer=[],
    )

    # a new input to decide if we use past state or not
    enable_cache_input = onnx.helper.make_tensor_value_info(
        name="enable_cache", elem_type=onnx.TensorProto.BOOL, shape=[1]
    )

    if_node = onnx.helper.make_node(
        op_type="If",
        inputs=["enable_cache"],
        outputs=[o.name for o in list(model_no_cache.graph.output)],
        then_branch=graph_cache,
        else_branch=graph_no_cache,
    )

    # final model which can disable its cache
    if_graph_def: onnx.GraphProto = onnx.helper.make_graph(
        nodes=[if_node],
        name="if-model",
        inputs=list(model_cache.graph.input) + [enable_cache_input],
        outputs=list(model_no_cache.graph.output),
        initializer=list(model_no_cache.graph.initializer),
    )

    # serialization and cleaning
    model_if: ModelProto = onnx.helper.make_model(
        if_graph_def, producer_name="onnx-example", opset_imports=[onnx.helper.make_opsetid(onnx.defs.ONNX_DOMAIN, 13)]
    )
    save_onnx(proto=model_if, model_path=output_path)


def convert_bf16_to_fp32(bf16_data: bytes) -> bytes:
    """
    Convert bf16 byte array to fp32 byte array.

    :param bf16_data: an array of bf16 numbers (from numpy)
    :return: an array of fp32 numbers (from numpy)
    """
    data: np.ndarray = np.frombuffer(bf16_data, dtype=np.uint16)
    zeros: np.ndarray = np.zeros_like(data)
    to_concatenate = [zeros, data] if sys.byteorder == "little" else [data, zeros]
    fp32_data: np.ndarray = np.ascontiguousarray(np.stack(to_concatenate).transpose()).view(np.float32).squeeze()
    return fp32_data.tobytes()


def convert_fp32_to_bf16(fp32_data: bytes) -> bytes:
    """
    Convert fp32 byte array to bf16 byte array.

    :param fp32_data: an array of fp32 numbers (from numpy)
    :return: an array of bf16 numbers (from numpy)
    """
    data: np.ndarray = np.frombuffer(fp32_data, dtype=np.float32)
    # view() shows the data in another format WITHOUT performing any conversion
    # therefore we just (virtually) split each number into two 16-bit parts
    fake_int16_view = data.view(dtype=np.int16)
    # we keep only one half of each, which part will depend of the OS (little or big endian)
    np_bfp16 = fake_int16_view[1::2] if sys.byteorder == "little" else fake_int16_view[0::2]
    return np_bfp16.tobytes()


def patch_constant_node_bf16(model: ModelProto) -> ModelProto:
    """
    Patch ONNX graph to convert ConstantOfShape operators using bf16 to fp32 + cast.
    ConstantOfShape in bf16 is not supported by ONNX.
    :param model: original ONNX model
    :return: ONNX model patched
    """
    model = copy.deepcopy(model)
    all_nodes = dict()
    for node in model.graph.node:
        for name in node.input:
            all_nodes[name] = node

    graph_index = 0
    while graph_index < len(model.graph.node):
        current_node = model.graph.node[graph_index]
        bfloat16 = onnx.TensorProto.BFLOAT16
        if current_node.op_type == "ConstantOfShape" and current_node.attribute[0].t.data_type == bfloat16:
            # change constant type to float
            current_node.attribute[0].t.data_type = onnx.TensorProto.FLOAT
            # current_node.attribute[0].t.raw_data = np.array(0, dtype=np.float32).tobytes()
            current_node.attribute[0].t.raw_data = convert_bf16_to_fp32(bf16_data=current_node.attribute[0].t.raw_data)
            # a = np.frombuffer(node.attribute[0].t.raw_data, dtype=np.uint16)
            # a.astype(np.float32)
            original_output_name = current_node.output[0]
            new_output_name = f"{original_output_name}_float"
            # node.output[0] = output_name
            cast_node: onnx.NodeProto = onnx.helper.make_node(
                op_type="Cast",
                inputs=[current_node.output[0]],
                outputs=[new_output_name],
                name=new_output_name,
                to=bfloat16,
            )
            model.graph.node.insert(graph_index + 1, cast_node)
            graph_index += 1  # skip the inserted node
            next_node = all_nodes[current_node.output[0]]
            for index, input_name in enumerate(next_node.input):
                if input_name == original_output_name:
                    next_node.input[index] = new_output_name
        graph_index += 1  # next node
    return model
