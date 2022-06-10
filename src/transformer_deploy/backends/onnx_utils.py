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
from pathlib import Path

import onnx
from onnx import ModelProto
from onnx.shape_inference import infer_shapes_path


def save_onnx(proto: onnx.ModelProto, model_path: str) -> None:
    """
    Save Onnx file with external data support when required.

    :param proto: Onnx model
    :param model_path: output path
    """
    # protobuff doesn't support files > 2Gb, in this case, weights are stored in another binary file
    save_external_data: bool = proto.ByteSize() > 2 * 1024**3
    filename = Path(model_path).name
    onnx.save_model(
        proto=proto,
        f=model_path,
        save_as_external_data=save_external_data,
        all_tensors_to_one_file=True,
        location=filename + ".data",
    )
    infer_shapes_path(model_path=model_path, output_path=model_path)


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
    for node_cache in model_cache.graph.initializer:
        found = False
        for node_no_cache in model_no_cache.graph.initializer:
            if node_cache.raw_data == node_no_cache.raw_data:
                found = True
                mapping_initializer_cache_to_no_cache[node_cache.name] = node_no_cache.name
                break
        if not found:
            node_cache.name = prefix_cache + node_cache.name
            to_add.append(node_cache)
            mapping_initializer_cache_to_no_cache[node_cache.name] = node_cache.name

    model_no_cache.graph.initializer.extend(to_add)
    # I/O model names should not be prefixed
    model_io_names = [n.name for n in list(model_cache.graph.input) + list(model_cache.graph.output)]

    # replace pointers to duplicated weights to their deduplicated version
    for node in model_cache.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in model_io_names:
                continue
            node.input[index] = mapping_initializer_cache_to_no_cache.get(input_name, prefix_cache + input_name)
        for index, output_name in enumerate(node.output):
            if output_name in model_io_names:
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
