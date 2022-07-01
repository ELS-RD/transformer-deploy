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
from queue import Queue
from typing import Dict, List

import numpy as np
import onnx
from onnx import NodeProto
from onnxruntime.transformers.onnx_model import OnnxModel

from transformer_deploy.backends.onnx_utils import merge_autoregressive_model_graphs, save_onnx
from transformer_deploy.backends.ort_utils import add_output_nodes, create_model_for_provider


# TODO test avec tiny GPT-2
dec_cache_model_path = "./demo/generative-model/test-dec-cache/model.onnx"
dec_no_cache_model_path = "./demo/generative-model/test-dec-no-cache/model.onnx"

dec_cache_model: onnx.ModelProto = onnx.load_model(f=dec_cache_model_path, load_external_data=False)
dec_no_cache_model: onnx.ModelProto = onnx.load_model(f=dec_no_cache_model_path, load_external_data=False)
assert len(dec_cache_model.graph.output) == len(dec_no_cache_model.graph.output)
original_nb_output_nodes = len(dec_cache_model.graph.output)

dec_cache_model_fp32_all_nodes = add_output_nodes(model=dec_cache_model)
dec_cache_model_fp32_all_nodes_path = dec_cache_model_path + "_all_nodes.onnx"
save_onnx(proto=dec_cache_model_fp32_all_nodes, model_path=dec_cache_model_fp32_all_nodes_path)
# _ = create_model_for_provider(dec_cache_model_fp32_all_nodes_path, "CUDAExecutionProvider", log_severity=3)

# reload after shape inference
dec_cache_model_fp32_all_nodes = onnx.load_model(f=dec_cache_model_fp32_all_nodes_path, load_external_data=False)

ort_np_type_mapping = {
    onnx.TensorProto.FLOAT: float,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.BOOL: bool,
}


# If node requires that the 2 models merged have the exact same number/type of output nodes
# Above we added many output nodes to the model with cache support...
# ... we need to add fake output nodes to the other decoder model.
original_output_nodes = {item.name: item for item in dec_no_cache_model.graph.output}


nb_outputs_to_create = len(dec_cache_model_fp32_all_nodes.graph.output)
nodes_to_be_added = list()

for i in range(nb_outputs_to_create):
    node_name = dec_cache_model_fp32_all_nodes.graph.output[i].name
    # if the output node is shared (aka its a real model output), then just add its name to the list of output nodes
    if node_name in original_output_nodes:
        original_output = original_output_nodes[
            node_name
        ]  # TODO onnx.helper.make_empty_tensor_value_info(name=node_name)
        nodes_to_be_added.append(original_output)
    else:
        fake_node_name = f"output_{node_name}"
        fake_node_ort_type = dec_cache_model_fp32_all_nodes.graph.output[i].type.tensor_type.elem_type
        fake_node_np_type = ort_np_type_mapping[fake_node_ort_type]
        fake_data = np.array([0.0], dtype=fake_node_np_type)
        fake_node = onnx.helper.make_node(
            op_type="Constant",
            inputs=[],
            outputs=[fake_node_name],
            value=onnx.helper.make_tensor(
                name=f"{fake_node_name}_const_tensor",
                data_type=fake_node_ort_type,
                dims=fake_data.shape,
                vals=fake_data.flatten(),
            ),
            name=fake_node_name,
        )

        dec_no_cache_model.graph.node.append(fake_node)
        nodes_to_be_added.append(onnx.helper.make_empty_tensor_value_info(name=fake_node_name))


dec_no_cache_model.graph.ClearField("output")
dec_no_cache_model.graph.output.extend(nodes_to_be_added)
# OnnxModel.graph_topological_sort(dec_no_cache_model.graph)

dec_no_cache_model_fp32_all_nodes_path = dec_no_cache_model_path + "_all_nodes.onnx"
save_onnx(proto=dec_no_cache_model, model_path=dec_no_cache_model_fp32_all_nodes_path)

# now that each model has the same number of output nodes, we can merge them!
merge_autoregressive_model_graphs(
    model_cache_path=dec_cache_model_fp32_all_nodes_path,
    model_no_cache_path=dec_no_cache_model_fp32_all_nodes_path,
    output_path="whatever.onnx",
)


def add_q(items: List[onnx.NodeProto]) -> None:
    for item in items:
        nodes.put(item=item)


_ = create_model_for_provider("whatever.onnx", "CUDAExecutionProvider", log_severity=3)

model = onnx.load_model("whatever.onnx", load_external_data=False)
nodes = Queue()
add_q(items=model.graph.node)

counter = 0
while not nodes.empty():
    node: NodeProto = nodes.get()
    if node.op_type == "If":
        add_q(items=node.attribute[0].g.node)
        add_q(items=node.attribute[1].g.node)
    if "cache_node_Identity_1" == node.name:  # 'onnx::Concat_1329' "cache_node_Identity_1"
        print(node)
        print(counter)
        print("---")
    counter += 1


# 'onnx::Concat_1329' is the output of a node in cache model with all nodes, in position 42
# input: "onnx::Concat_1247"
# output: "onnx::Concat_1329"
# name: "Identity_42"
# op_type: "Identity"
# 42
# in merge model, it's still at the same position (42, or 1794 in the merged model, 1751 first positions are for no cache model)
# each model has 1751 nodes
# cache_node_Identity_1 is in position 1753 in merged model

# -----
# in dec_cache_model_fp32_all_nodes_path:
# input: "onnx::Concat_1241"
# output: "onnx::Concat_1389"
# name: "Identity_1"
# op_type: "Identity"
# ---
# in "whatever.onnx"
# input: "onnx::Concat_1329"
# output: "onnx::Concat_1389"
# name: "cache_node_Identity_1"
# op_type: "Identity"
# ----
# input has changed, may be due to the deduplication process in the merge task
