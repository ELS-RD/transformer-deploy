import copy
from collections import OrderedDict, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import onnx
import torch
from onnx import GraphProto, ModelProto, NodeProto, shape_inference
from onnxruntime import InferenceSession
from onnxruntime.transformers.float16 import convert_float_to_float16
from torch.nn import Linear
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, TensorType
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding, optimize_onnx
from transformer_deploy.backends.pytorch_utils import convert_to_onnx


# class ExportT5(torch.nn.Module):
#     def __init__(self, decoder: T5Stack, lm_head: Linear):
#         super(ExportT5, self).__init__()
#         self.decoder = decoder
#         self.lm_head = lm_head
#
#     def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor):
#         out_dec = self.decoder.forward(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
#         # Rescale output before projecting on vocab
#         out_dec = out_dec["last_hidden_state"] * (model_pytorch.model_dim**-0.5)
#         out_lm = self.lm_head(out_dec)
#         return out_lm
#
#
# model_name = "t5-small"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model_pytorch: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# model_pytorch = model_pytorch.eval()
# model_pytorch = model_pytorch.to("cuda")
# input_ids: torch.Tensor = tokenizer("Studies show that", return_tensors=TensorType.PYTORCH).input_ids
# input_ids = input_ids.to("cuda")
#
# out_enc: BaseModelOutputWithPastAndCrossAttentions = model_pytorch.encoder(input_ids=input_ids)
# convert_to_onnx(
#     model_pytorch=model_pytorch.encoder,
#     output_path="test-enc.onnx",
#     inputs_pytorch={"input_ids": input_ids},
#     var_output_seq=True,
#     quantization=False,
# )
# optimize_onnx(
#     onnx_path="test-enc.onnx", onnx_optim_model_path="test-enc-opt.onnx", architecture="bert", use_cuda=True, fp16=True
# )
#
# enc_onnx = create_model_for_provider("test-enc-opt.onnx", "CUDAExecutionProvider")
# enc_onnx_out = inference_onnx_binding(
#     model_onnx=enc_onnx,
#     inputs={"input_ids": input_ids},
#     device=input_ids.device.type,
#     output_shape=tuple(input_ids.shape) + (model_pytorch.encoder.config.d_model,),
# )["output"]
# assert np.allclose(enc_onnx_out.detach().cpu().numpy(), out_enc.last_hidden_state.detach().cpu().numpy(), atol=1e-2)
#
# # https://github.com/microsoft/onnxruntime/issues/1455#issuecomment-979901463
# # add all intermediate outputs to onnx net
# org_outputs = [x.name for x in enc_onnx.get_outputs()]

# -------------------------

model_onnx: ModelProto = onnx.load("test-enc.onnx")
model_onnx_fp16 = convert_float_to_float16(model=onnx.load("test-enc.onnx"))


# https://en.wikipedia.org/wiki/Machine_epsilon
# 2**-24
def convert_np_to_float16(np_array: np.ndarray, min_positive_val: float = 5.96e-08, max_finite_val: float = 65504.0):
    def between(a, b, c):
        return np.logical_and(a < b, b < c)
    # minimum pos value
    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    # maximum neg value
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float('inf')), max_finite_val, np_array)
    np_array = np.where(between(float('-inf'), np_array, -max_finite_val), -max_finite_val, np_array)
    return np.float16(np_array)


initializer_onnx = dict()
for n in model_onnx.graph.initializer:
    type_tensor: str = n.DataType.Name(n.data_type).lower()
    if type_tensor == "float":
        type_tensor = "float32"
    initializer_onnx[n.name] = np.frombuffer(n.raw_data, dtype=type_tensor).reshape(n.dims)


def get_output(model: ModelProto, shape: Tuple[int]) -> Dict[str, np.ndarray]:
    model = copy.deepcopy(model)
    output_nodes = list()
    for n in model.graph.node:
        for output_name in n.output:
            output_nodes.append(onnx.ValueInfoProto(name=output_name))
    model.graph.output.extend(output_nodes)
    ort_model = create_model_for_provider(model.SerializeToString(), "CPUExecutionProvider")
    outputs_names = [x.name for x in ort_model.get_outputs()]
    ort_outs = ort_model.run(outputs_names, {"input_ids": np.ones(shape=shape, dtype=np.int32)})
    return OrderedDict(zip(outputs_names, ort_outs))


outputs = get_output(model=model_onnx, shape=(5, 10))
outputs_fp16 = get_output(model=model_onnx_fp16, shape=(5, 10))
all_input_tensors = outputs | initializer_onnx
all_input_tensors["input_ids"] = np.ones(shape=(1, 4), dtype=np.int32)
all_input_tensors_fp16 = dict()
for k, v in all_input_tensors.items():
    if v.dtype == np.float32:
        # v = convert_np_to_float16(v)
        v = np.float16(v)
    all_input_tensors_fp16[k] = v


def get_onnx_type(np_type: str) -> int:
    if np_type == "int64":
        return onnx.TensorProto.INT64
    elif np_type == "int32":
        return onnx.TensorProto.INT32
    elif np_type == "float32":
        return onnx.TensorProto.FLOAT
    elif np_type == "float16":
        return onnx.TensorProto.FLOAT16
    elif np_type == "bool":
        return onnx.TensorProto.BOOL
    else:
        raise Exception(f"unknown type {np_type}")


def get_graph_io(node_names: List[str], exclude: Set[str], graph_inputs: Dict[str, np.ndarray]) -> List[NodeProto]:
    io: List[NodeProto] = list()
    for node_name in node_names:  # type: str
        if node_name in exclude:
            continue
        node = onnx.helper.make_tensor_value_info(
            name=node_name,
            elem_type=get_onnx_type(np_type=graph_inputs[node_name].dtype.name),
            shape=graph_inputs[node_name].shape,
        )
        io.append(node)
    return io


def make_graph(nodes: List[NodeProto], graph_inputs: Dict[str, np.ndarray], name: str) -> GraphProto:
    input_names: List[str] = [node_name for node in nodes for node_name in node.input]
    output_names: List[str] = [node.output[0] for node in nodes]

    return onnx.helper.make_graph(
        nodes=nodes,
        name=name,
        inputs=get_graph_io(node_names=input_names, exclude=set(output_names), graph_inputs=graph_inputs),
        outputs=get_graph_io(node_names=output_names, exclude=set(input_names), graph_inputs=graph_inputs),
        initializer=[],
    )


def make_model(nodes: List[NodeProto], graph_inputs: Dict[str, np.ndarray], name: str = "test-model") -> InferenceSession:
    graph_def: GraphProto = make_graph(nodes=nodes, graph_inputs=graph_inputs, name=name)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13
    model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(model_def)
    return create_model_for_provider(model_def.SerializeToString(), "CPUExecutionProvider")


def check_model(model: InferenceSession, graph_inputs: Dict[str, np.ndarray], atol: float) -> List[np.ndarray]:
    # inputs = {node.name: graph_inputs[node.name] for node in model.get_inputs()}
    inputs: Dict[str, np.ndarray] = dict()
    for input_node in model.get_inputs():
        inputs[input_node.name] = graph_inputs[input_node.name]
        if inputs[input_node.name].shape == ():
            inputs[input_node.name] = inputs[input_node.name].reshape((1, ))

    output_names = [output_node.name for output_node in model.get_outputs()]
    preds = test_model.run(
        output_names=None,
        input_feed=inputs,
    )

    assert len(output_names) == len(preds)
    for name, pred in zip(output_names, preds):
        pred = np.nan_to_num(pred)
        assert np.allclose(pred, graph_inputs[name], atol=atol), f"error on {name}"
    return preds


all_input_tensors["input_ids"] = np.ones((5, 10), dtype=np.int32)
all_input_tensors_fp16["input_ids"] = np.ones((5, 10), dtype=np.int32)
for node in model_onnx.graph.node:
    test_model = make_model(nodes=[node], graph_inputs=all_input_tensors)
    out_fp32 = check_model(model=test_model, graph_inputs=all_input_tensors, atol=1e-2)
    if node.op_type == "Cast":
        continue
    for index_attr in range(len(node.attribute)):
        if node.attribute[index_attr].t.data_type == onnx.TensorProto.FLOAT:
            node.attribute[index_attr].t.data_type = onnx.TensorProto.FLOAT16
            data: np.ndarray = np.frombuffer(node.attribute[index_attr].t.raw_data, dtype=np.float32).reshape(node.attribute[index_attr].t.dims)
            data = data.astype(np.float16)
            node.attribute[index_attr].t.raw_data = data.tobytes()

    test_model = make_model(nodes=[node], graph_inputs=all_input_tensors_fp16)
    out_fp16 = check_model(model=test_model, graph_inputs=all_input_tensors_fp16, atol=1e-2)

    for a, b in zip(out_fp32, out_fp16):
        assert np.allclose(a, b, atol=1e-2)


# onnx_graph: Dict[str, Set[str]] = defaultdict(set)
# for n in model_onnx.graph.node:  # type: NodeProto
#     # print(n.name)
#     # print(n.op_type)
#     # print(n.input)
#     # print(n.output)
#     # print(n.attribute)
#     # print("----")
#     assert len(n.output) == 1
#     output_node = n.output[0]
#     for i in n.input:
#         onnx_graph[i].add(output_node)


# TODO convert non optimized model to FP16 and compare node 84 associated output
# https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/float16.py


# def search(graph: Dict[str, Set[str]], root: str, visited: Dict[str, None]):
#     if root not in graph:
#         return
#     nodes = graph[root]
#     print("++" + root)
#     visited[root] = None
#     for next_node in nodes:
#         if next_node not in visited:
#             search(graph=graph, root=next_node, visited=visited)
#         else:
#             print("--" + root)
#
#
# known = dict()
# search(graph=onnx_graph, root="input_ids", visited=known)

