import onnx
from onnx import helper, ValueInfoProto, NodeProto, ModelProto
from onnx import TensorProto, GraphProto
import numpy as np
from onnxruntime import InferenceSession

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

# I/O
X: ValueInfoProto = helper.make_tensor_value_info('X', TensorProto.FLOAT, shape=["axis1", "axis2"])
Y: ValueInfoProto = helper.make_tensor_value_info('Y', TensorProto.FLOAT, shape=["axis1", "axis2"])
OUT: ValueInfoProto = helper.make_tensor_value_info('OUT', TensorProto.FLOAT, shape=[1, 4])
Z: ValueInfoProto = helper.make_tensor_value_info('Z', TensorProto.FLOAT, shape=[1, 4])

# Create a node (NodeProto)
add_1_node: NodeProto = helper.make_node(
    op_type='Add',
    inputs=['X', 'Tensor'],
    outputs=['add1'],
    name="first_add",
)

add_2_node: NodeProto = helper.make_node(
    op_type='Add',
    inputs=['add1', 'Tensor'],
    outputs=['OUT'],
    name="second_add",
)

graph_add: GraphProto = helper.make_graph(
    nodes=[add_1_node, add_2_node],
    name="add-model",
    inputs=[],
    outputs=[OUT],
    initializer=[],
)

# Create a node (NodeProto)
sub_1_node: NodeProto = helper.make_node(
    op_type='Sub',
    inputs=['X', 'Tensor'],
    outputs=['sub1'],
    name="first_sub",
)

sub_2_node: NodeProto = helper.make_node(
    op_type='Sub',
    inputs=['sub1', 'Tensor'],
    outputs=['OUT'],
    name="second_sub",
)

graph_sub: GraphProto = helper.make_graph(
    nodes=[sub_1_node, sub_2_node],
    name="sub-model",
    inputs=[],
    outputs=[OUT],
    initializer=[],
)

retrieve_shape: NodeProto = helper.make_node(
    op_type='Shape',
    inputs=['Y'],
    outputs=['shape_2'],
)

gather_def: NodeProto = helper.make_node(
    op_type='Gather',
    inputs=['shape_2', 'gather_dim_index'],
    outputs=['gather1'],
)

equal_def: NodeProto = helper.make_node(
    op_type='Equal',
    inputs=['gather1', 'expected_val'],
    outputs=['eq_vec'],
)

squeeze_def: NodeProto = helper.make_node(
    op_type='Squeeze',
    inputs=['eq_vec', 'squeeze_dim'],
    outputs=['if_cond'],
)

if_node = onnx.helper.make_node(
    'If',
    inputs=['if_cond'],
    outputs=['Z'],
    then_branch=graph_add,
    else_branch=graph_sub
)

# vals are flatten values
init_tensor = helper.make_tensor('Tensor', dims=(1, 4), vals=[1., 2., 3., 4.], data_type=TensorProto.FLOAT)
gather_dim_index = helper.make_tensor('gather_dim_index', dims=(1, ), vals=[1], data_type=TensorProto.INT32)
expected_val = helper.make_tensor('expected_val', dims=(1, ), vals=[4], data_type=TensorProto.INT64)
squeeze_dim = helper.make_tensor('squeeze_dim', dims=(1, ), vals=[0], data_type=TensorProto.INT64)


# Create the graph (GraphProto)
if_graph_def: GraphProto = helper.make_graph(
    nodes=[retrieve_shape, gather_def, equal_def, squeeze_def, if_node],
    name="if-model",
    inputs=[X, Y],
    outputs=[Z],
    initializer=[gather_dim_index, expected_val, squeeze_dim, init_tensor],
)

model_def: ModelProto = helper.make_model(if_graph_def, producer_name='onnx-example')

print('The graph in model:\n{}'.format(model_def.graph))
onnx.checker.check_model(model_def)

print("model loading")
ort_model = InferenceSession(model_def.SerializeToString(), providers=["CPUExecutionProvider"])

print(ort_model.run(None, {'X': np.ones((1, 4), dtype=np.float32), 'Y': np.ones((1, 4), dtype=np.float32)}))
# https://github.com/onnx/onnx/tree/main/onnx/examples
