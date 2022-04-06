import copy
from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import onnx
import tensorrt as trt
import torch
from onnx import ModelProto, NodeProto
from tensorrt import ICudaEngine, ILayer, INetworkDefinition, LayerType, Logger, Runtime
from torch.nn import Linear
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast, TensorType
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding
from transformer_deploy.backends.pytorch_utils import convert_to_onnx
from transformer_deploy.backends.trt_utils import build_engine, load_engine, save_engine


class ExportT5(torch.nn.Module):
    def __init__(self, decoder: T5Stack, lm_head: Linear):
        super(ExportT5, self).__init__()
        self.decoder = decoder
        self.lm_head = lm_head

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor):
        out_dec = self.decoder.forward(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
        # Rescale output before projecting on vocab
        out_dec = out_dec["last_hidden_state"] * (model_pytorch.model_dim**-0.5)
        out_lm = self.lm_head(out_dec)
        return out_lm


model_name = "t5-small"
tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(model_name)
model_pytorch: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model_pytorch = model_pytorch.eval()
model_pytorch = model_pytorch.to("cuda")
input_ids: torch.Tensor = tokenizer("Studies show that", return_tensors=TensorType.PYTORCH).input_ids
input_ids = input_ids.to("cuda")

out_enc: BaseModelOutputWithPastAndCrossAttentions = model_pytorch.encoder(input_ids=input_ids)
convert_to_onnx(
    model_pytorch=model_pytorch.encoder,
    output_path="test-enc.onnx",
    inputs_pytorch={"input_ids": input_ids},
    var_output_seq=True,
    quantization=False,
)

enc_onnx = create_model_for_provider("test-enc.onnx", "CUDAExecutionProvider")
enc_onnx_out = inference_onnx_binding(
    model_onnx=enc_onnx,
    inputs={"input_ids": input_ids},
    device=input_ids.device.type,
    output_shape=tuple(input_ids.shape) + (model_pytorch.encoder.config.d_model,),
)["output"]
assert np.allclose(enc_onnx_out.detach().cpu().numpy(), out_enc.last_hidden_state.detach().cpu().numpy(), atol=1e-2)

# https://github.com/microsoft/onnxruntime/issues/1455#issuecomment-979901463
# add all intermediate outputs to onnx net
org_outputs = [x.name for x in enc_onnx.get_outputs()]


# https://en.wikipedia.org/wiki/Machine_epsilon
# 2**-24
# from onnxruntime package
def convert_fp16(np_array: np.ndarray, min_positive_val: float = 5.96e-08, max_finite_val: float = 65504.0):
    def between(a, b, c):
        return np.logical_and(a < b, b < c)

    # minimum pos value
    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    # maximum neg value
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float("inf")), max_finite_val, np_array)
    np_array = np.where(between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array)
    return np.float16(np_array)


def add_output_nodes(model: ModelProto) -> ModelProto:
    model = copy.deepcopy(model)
    output_nodes = list()
    for n in model.graph.node:
        for output_name in n.output:
            output_nodes.append(onnx.ValueInfoProto(name=output_name))
    model.graph.output.extend(output_nodes)
    return model


def get_all_outputs(model: ModelProto, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    outputs_names = [x.name for x in model.get_outputs()]
    ort_outs = model.run(output_names=outputs_names, input_feed=inputs)
    return dict(zip(outputs_names, ort_outs))


# https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/float16.py
def find_node_fp32(graph: Dict[str, Set[str]], output_nodes: Dict[str, np.ndarray]) -> List[str]:
    keep_fp32 = list()
    for k, v in output_nodes.items():
        if v.dtype != np.float32:
            continue
        if np.max(v) > np.finfo(np.float16).max or np.min(v) < np.finfo(np.float16).min:
            keep_fp32 += [n for n in graph[k]]
    return keep_fp32


def get_fix_fp16_network_func(keep_fp32: List[str]) -> Callable[[INetworkDefinition], INetworkDefinition]:
    def f(network_definition: INetworkDefinition) -> INetworkDefinition:
        for layer_index in range(network_definition.num_layers - 1):
            layer: ILayer = network_definition.get_layer(layer_index)
            # next layer should take FP16 as input
            next_layer: ILayer = network_definition.get_layer(layer_index + 1)

            if layer.name in keep_fp32 and next_layer.type != LayerType.IDENTITY:
                layer.precision = trt.DataType.FLOAT
                layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
                # identity function is mainly used for casting
                # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Layers.html#iidentitylayer
                if next_layer.type != LayerType.IDENTITY:
                    next_layer.precision = trt.DataType.FLOAT
                    # next_layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)

        return network_definition

    return f


def get_adjency_list(model: ModelProto) -> Dict[str, Set[str]]:
    adj_dict: Dict[str, Set[str]] = defaultdict(set)
    for n in model.graph.node:  # type: NodeProto
        assert len(n.output) == 1
        output_node = n.output[0]
        adj_dict[output_node].add(n.name)
    return adj_dict


model_onnx: ModelProto = onnx.load("test-enc.onnx")
model_onnx_all_nodes = add_output_nodes(model=model_onnx)
onnx_graph: Dict[str, Set[str]] = get_adjency_list(model=model_onnx)
ort_model_all_nodes = create_model_for_provider(model_onnx_all_nodes.SerializeToString(), "CUDAExecutionProvider")

all_outputs = list()
for _ in range(200):
    # use info from tokenizer size and max shape provided through the command line
    inputs = {"input_ids": np.random.randint(low=0, high=32100, size=(32, 512), dtype=np.int32)}
    outputs: Dict[str, np.ndarray] = get_all_outputs(model=ort_model_all_nodes, inputs=inputs)
    keep_node_io = find_node_fp32(graph=onnx_graph, output_nodes=outputs)
    all_outputs.append(keep_node_io)

# TODO make it a function
counter = defaultdict(lambda: 0)
for o in all_outputs:
    for node_name in o:
        counter[node_name] += 1


trt_logger: Logger = trt.Logger(trt.Logger.INFO)
runtime: Runtime = trt.Runtime(trt_logger)
engine: ICudaEngine = build_engine(
    runtime=runtime,
    onnx_file_path="test-enc.onnx",
    logger=trt_logger,
    min_shape=(1, 5),
    optimal_shape=(1, 5),
    max_shape=(1, 5),
    workspace_size=12 * 1024 * 1024,
    fp16=True,
    int8=False,
    fp16_fix=get_fix_fp16_network_func(keep_fp32=keep_node_io),
)
save_engine(engine=engine, engine_file_path="test-enc.plan")

tensorrt_model: Callable[[Dict[str, torch.Tensor]], torch.Tensor] = load_engine(
    runtime=runtime, engine_file_path="test-enc.plan"
)

print(tensorrt_model({"input_ids": torch.ones((1, 5), dtype=torch.int32)}))

model_ort = create_model_for_provider(model_onnx.SerializeToString(), "CPUExecutionProvider")
original_output_onnx = model_ort.run(None, {"input_ids": np.ones((1, 5), dtype=np.int32)})
print(original_output_onnx)
