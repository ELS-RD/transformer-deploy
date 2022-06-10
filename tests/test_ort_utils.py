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

import onnx
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, helper

from transformer_deploy.backends.ort_utils import get_io_to_node_mapping


def get_onnx_if() -> ModelProto:
    """
    Generate an Onnx model with an If node
    :return: Onnx model
    """
    # I/O
    input_x: ValueInfoProto = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape=["axis1", "axis2"])
    input_y: ValueInfoProto = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape=["axis1", "axis2"])
    out: ValueInfoProto = helper.make_tensor_value_info("OUT", TensorProto.FLOAT, shape=[1, 4])
    outptu_z: ValueInfoProto = helper.make_tensor_value_info("Z", TensorProto.FLOAT, shape=[1, 4])

    # Create a node (NodeProto)
    add_1_node: NodeProto = helper.make_node(
        op_type="Add",
        inputs=["X", "Tensor"],
        outputs=["add1"],
        name="first_add",
    )

    add_2_node: NodeProto = helper.make_node(
        op_type="Add",
        inputs=["add1", "Tensor"],
        outputs=["OUT"],
        name="second_add",
    )

    graph_add: GraphProto = helper.make_graph(
        nodes=[add_1_node, add_2_node],
        name="add-model",
        inputs=[],
        outputs=[out],
        initializer=[],
    )

    # Create a node (NodeProto)
    sub_1_node: NodeProto = helper.make_node(
        op_type="Sub",
        inputs=["X", "Tensor"],
        outputs=["sub1"],
        name="first_sub",
    )

    sub_2_node: NodeProto = helper.make_node(
        op_type="Sub",
        inputs=["sub1", "Tensor"],
        outputs=["OUT"],
        name="second_sub",
    )

    graph_sub: GraphProto = helper.make_graph(
        nodes=[sub_1_node, sub_2_node],
        name="sub-model",
        inputs=[],
        outputs=[out],
        initializer=[],
    )

    retrieve_shape: NodeProto = helper.make_node(
        op_type="Shape",
        inputs=["Y"],
        outputs=["shape_2"],
        name="retrieve_shape",
    )

    gather_def: NodeProto = helper.make_node(
        op_type="Gather",
        inputs=["shape_2", "gather_dim_index"],
        outputs=["gather1"],
        name="gather_def",
    )

    equal_def: NodeProto = helper.make_node(
        op_type="Equal",
        inputs=["gather1", "expected_val"],
        outputs=["eq_vec"],
        name="equal_def",
    )

    squeeze_def: NodeProto = helper.make_node(
        op_type="Squeeze",
        inputs=["eq_vec", "squeeze_dim"],
        outputs=["if_cond"],
        name="squeeze_def",
    )

    if_node = onnx.helper.make_node(
        "If", inputs=["if_cond"], outputs=["Z"], then_branch=graph_add, else_branch=graph_sub, name="if_node"
    )

    # vals are flatten values
    init_tensor = helper.make_tensor("Tensor", dims=(1, 4), vals=[1.0, 2.0, 3.0, 4.0], data_type=TensorProto.FLOAT)
    gather_dim_index = helper.make_tensor("gather_dim_index", dims=(1,), vals=[1], data_type=TensorProto.INT32)
    expected_val = helper.make_tensor("expected_val", dims=(1,), vals=[4], data_type=TensorProto.INT64)
    squeeze_dim = helper.make_tensor("squeeze_dim", dims=(1,), vals=[0], data_type=TensorProto.INT64)

    # Create the graph (GraphProto)
    if_graph_def: GraphProto = helper.make_graph(
        nodes=[retrieve_shape, gather_def, equal_def, squeeze_def, if_node],
        name="if-model",
        inputs=[input_x, input_y],
        outputs=[outptu_z],
        initializer=[gather_dim_index, expected_val, squeeze_dim, init_tensor],
    )

    model_def: ModelProto = helper.make_model(if_graph_def, producer_name="onnx-example")
    onnx.checker.check_model(model_def)
    return model_def


def test_io_mapping():
    if_model = get_onnx_if()
    inputs, outputs = get_io_to_node_mapping(onnx_model=if_model)
    assert len(inputs) == 12
    assert len(outputs) == 8
    # check k, v are all defined
    for k, v in list(inputs.items()) + list(outputs.items()):
        assert k
        assert v
