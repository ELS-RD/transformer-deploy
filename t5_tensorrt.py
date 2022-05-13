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

from time import time
from typing import Optional, Tuple

import numpy as np
import onnx
import tensorrt as trt
import torch
from onnx import GraphProto, ModelProto, helper
from tensorrt import ICudaEngine, Logger, Runtime
from torch.nn import Linear
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, TensorType
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding
from transformer_deploy.backends.pytorch_utils import convert_to_onnx
from transformer_deploy.backends.trt_utils import TensorRTShape, build_engine, load_engine, save_engine


model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids: torch.Tensor = tokenizer(
    "translate English to French: This model is now very fast!", return_tensors=TensorType.PYTORCH
).input_ids
input_ids = input_ids.to("cuda")
model: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.eval()
model = model.to("cuda")
model.config.use_cache = True
out_enc: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=input_ids)
out_full: Seq2SeqLMOutput = model(input_ids=input_ids, decoder_input_ids=input_ids)
num_layers = model.config.num_layers
model = model.to("cuda")


def are_equal(a: torch.Tensor, b: torch.Tensor, atol: float = 5e-1) -> None:
    assert np.allclose(a=a.detach().cpu().numpy(), b=b.detach().cpu().numpy(), atol=atol), f"{a}\n\nVS\n\n{b}"


convert_to_onnx(
    model_pytorch=model.encoder,
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
)["output"]

are_equal(a=enc_onnx_out, b=out_enc.last_hidden_state)


class ExportT5(torch.nn.Module):
    def __init__(self, decoder: T5Stack, lm_head: Linear):
        super(ExportT5, self).__init__()
        self.decoder = decoder
        self.lm_head = lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        final_seq_len: Optional[torch.Tensor],
        past_key_values: Tuple = None,
    ):
        out_dec = self.decoder.forward(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values
        )
        # Rescale output before projecting on vocab
        out_dec["last_hidden_state"] = out_dec["last_hidden_state"] * (model.model_dim**-0.5)
        out_dec["last_hidden_state"] = self.lm_head(out_dec["last_hidden_state"])
        out_dec["past_key_values"] = list(out_dec["past_key_values"])
        for i, layer_out in enumerate(out_dec["past_key_values"]):  # type: int, Tuple
            assert len(layer_out) == 4
            layer_out_l = list(layer_out)
            for j, l in enumerate(layer_out):  # type: int, torch.Tensor
                if j <= 1:
                    layer_out_l[j] = l[:, :, : final_seq_len[0], :]
                else:
                    layer_out_l[j] = l
            out_dec["past_key_values"][i] = tuple(layer_out_l)
        out_dec["past_key_values"] = tuple(out_dec["past_key_values"])
        return out_dec


model.cuda()
model_decoder = ExportT5(decoder=model.decoder, lm_head=model.lm_head).eval()
out_model_export: torch.Tensor = model_decoder(
    input_ids=input_ids,
    encoder_hidden_states=out_enc.last_hidden_state,
    final_seq_len=torch.tensor([input_ids.shape[1]], dtype=torch.int32),
)

are_equal(a=out_model_export["last_hidden_state"], b=out_full.logits)


model_decoder.cuda()
# decoder output one step before
out_dec_pytorch = model_decoder(
    input_ids=input_ids[:, :-1],
    encoder_hidden_states=out_enc.last_hidden_state,
    final_seq_len=torch.tensor([1], dtype=torch.int32),
)

model_inputs = {
    "input_ids": input_ids[:, -1:].type(torch.int32),
    "encoder_hidden_states": out_enc.last_hidden_state,
    "past_key_values": out_dec_pytorch.past_key_values,
    "final_seq_len": torch.tensor([1], dtype=torch.int32),  # make it a 1 dim array
}

input_names = ["input_ids", "encoder_hidden_states", "final_seq_len"]

for i in range(num_layers):
    input_names.append(f"past_key_values.{i}.decoder.key")
    input_names.append(f"past_key_values.{i}.decoder.value")
    input_names.append(f"past_key_values.{i}.encoder.key")
    input_names.append(f"past_key_values.{i}.encoder.value")

output_names = ["logits"]

for i in range(num_layers):
    output_names.append(f"present.{i}.decoder.key")
    output_names.append(f"present.{i}.decoder.value")
    output_names.append(f"present.{i}.encoder.key")
    output_names.append(f"present.{i}.encoder.value")

dynamic_axis = {
    "input_ids": {0: "batch", 1: "decoder_sequence"},
    "encoder_hidden_states": {0: "batch", 1: "encoder_sequence_length"},
    "logits": {0: "batch", 1: "decoder_sequence"},
}


for i in range(num_layers):
    dynamic_axis[f"past_key_values.{i}.decoder.key"] = {0: "batch", 2: "past_decoder_sequence"}
    dynamic_axis[f"past_key_values.{i}.decoder.value"] = {0: "batch", 2: "past_decoder_sequence"}
    dynamic_axis[f"past_key_values.{i}.encoder.key"] = {0: "batch", 2: "encoder_sequence_length"}
    dynamic_axis[f"past_key_values.{i}.encoder.value"] = {0: "batch", 2: "encoder_sequence_length"}

    dynamic_axis[f"present.{i}.decoder.key"] = {0: "batch", 2: "decoder_sequence"}
    dynamic_axis[f"present.{i}.decoder.value"] = {0: "batch", 2: "decoder_sequence"}
    dynamic_axis[f"present.{i}.encoder.key"] = {0: "batch", 2: "encoder_sequence_length"}
    dynamic_axis[f"present.{i}.encoder.value"] = {0: "batch", 2: "encoder_sequence_length"}

with torch.no_grad():
    model.config.return_dict = True
    model.eval()

    # export can works with named args but the dict containing named args as to be last element of the args tuple
    torch.onnx.export(
        model_decoder,
        (model_inputs,),
        f="test-dec-cache.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axis,
        do_constant_folding=True,
        opset_version=13,
    )

model_inputs_no_cache = {
    "input_ids": input_ids.type(dtype=torch.int32),
    "encoder_hidden_states": out_enc.last_hidden_state,
    "final_seq_len": torch.tensor([input_ids.shape[1]], dtype=torch.int32),
}

with torch.no_grad():
    model.config.return_dict = True
    model.eval()

    # export can works with named args but the dict containing named args as to be last element of the args tuple
    torch.onnx.export(
        model_decoder,
        (model_inputs_no_cache,),
        f="test-dec-no-cache.onnx",
        input_names=list(model_inputs_no_cache.keys()),
        output_names=output_names,
        dynamic_axes={k: v for k, v in dynamic_axis.items() if "past_key_values" not in k},
        do_constant_folding=True,
        opset_version=13,
    )


_ = model_decoder.cpu()  # free cuda memory


onnx_model_no_cache_fp16 = onnx.load("test-dec-no-cache.onnx")
onnx_model_cache_fp16 = onnx.load("test-dec-cache.onnx")


assert len(onnx_model_cache_fp16.graph.output) == len(onnx_model_no_cache_fp16.graph.output)

final_output = list()
for node in onnx_model_cache_fp16.graph.output:
    new_output = onnx.helper.make_empty_tensor_value_info(node.name)
    new_output.CopyFrom(node)
    final_output.append(new_output)

final_node_names = [n.name for n in final_output]

for node in onnx_model_cache_fp16.graph.output:
    node.name += "-cache"

for node in onnx_model_cache_fp16.graph.node:
    assert len(node.output) == 1
    if node.output[0] in final_node_names:
        node.output[0] += "-cache"
    for idx, i in enumerate(node.input):
        if i in final_node_names:
            node.input[idx] += "-cache"

for node in onnx_model_no_cache_fp16.graph.output:
    node.name += "-no-cache"


for node in onnx_model_no_cache_fp16.graph.node:
    assert len(node.output) == 1
    if node.output[0] in final_node_names:
        node.output[0] += "-no-cache"
    for idx, i in enumerate(node.input):
        if i in final_node_names:
            node.input[idx] += "-no-cache"

onnx.checker.check_model(onnx_model_cache_fp16)
onnx.checker.check_model(onnx_model_no_cache_fp16)

prefix = "cache_node_"
mapping_initializer_cache_to_no_cache = dict()
to_add = list()
for node_cache in onnx_model_cache_fp16.graph.initializer:
    found = False
    for node_no_cache in onnx_model_no_cache_fp16.graph.initializer:
        if node_cache.raw_data == node_no_cache.raw_data:
            found = True
            mapping_initializer_cache_to_no_cache[node_cache.name] = node_no_cache.name
            break
    if not found:
        node_cache.name = prefix + node_cache.name
        to_add.append(node_cache)
        mapping_initializer_cache_to_no_cache[node_cache.name] = node_cache.name
        print(f"name: {node_cache.name} - size: {len(node_cache.raw_data)/1024:.2f}")

onnx_model_no_cache_fp16.graph.initializer.extend(to_add)
# I/O model names should not be prefixed
model_io_names = [
    n.name
    for n in list(onnx_model_cache_fp16.graph.input)
    + list(onnx_model_cache_fp16.graph.output)
    + list(onnx_model_no_cache_fp16.graph.input)
    + list(onnx_model_no_cache_fp16.graph.output)
]

for node in onnx_model_cache_fp16.graph.node:
    for index, input_name in enumerate(node.input):
        if input_name in model_io_names:
            continue
        node.input[index] = mapping_initializer_cache_to_no_cache.get(input_name, prefix + input_name)
    for index, output_name in enumerate(node.output):
        if output_name in model_io_names:
            continue
        node.output[index] = prefix + output_name
    node.name = prefix + node.name

prefix = "init_"
cache = dict()
for node in onnx_model_no_cache_fp16.graph.initializer:
    if node.name in model_io_names:
        new_name = prefix + node.name
        cache[node.name] = new_name
        node.name = new_name

for node in onnx_model_no_cache_fp16.graph.node:
    for input_index, n in enumerate(node.input):
        node.input[input_index] = cache.get(n, n)

# mandatory for subgraph in if/else node
assert len(onnx_model_cache_fp16.graph.output) == len(onnx_model_no_cache_fp16.graph.output)

graph_cache: onnx.GraphProto = onnx.helper.make_graph(
    nodes=list(onnx_model_cache_fp16.graph.node),
    name="graph-cache",
    inputs=[],
    outputs=list(onnx_model_cache_fp16.graph.output),
    initializer=[],
)

graph_no_cache: onnx.GraphProto = onnx.helper.make_graph(
    nodes=list(onnx_model_no_cache_fp16.graph.node),
    name="graph-no-cache",
    inputs=[],
    outputs=list(onnx_model_no_cache_fp16.graph.output),
    initializer=[],
)

enable_cache_input = onnx.helper.make_tensor_value_info(name="enable_cache", elem_type=onnx.TensorProto.BOOL, shape=[1])

if_node = onnx.helper.make_node(
    op_type="If",
    inputs=["enable_cache"],
    outputs=[o.name for o in final_output],
    then_branch=graph_cache,
    else_branch=graph_no_cache,
)

if_graph_def: GraphProto = helper.make_graph(
    nodes=[if_node],
    name="if-model",
    inputs=list(onnx_model_cache_fp16.graph.input) + [enable_cache_input],
    outputs=final_output,
    initializer=list(onnx_model_no_cache_fp16.graph.initializer),
)


model_def: ModelProto = helper.make_model(
    if_graph_def, producer_name="onnx-example", opset_imports=[helper.make_opsetid(onnx.defs.ONNX_DOMAIN, 13)]
)
onnx.save(model_def, "test-dec-if.onnx")


trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
runtime: Runtime = trt.Runtime(trt_logger)
trt_model_name = "trt-t5-dec.plan"

# 768 for base model, 512 for small, make it dependent from the Pytorch model configuration

shape, seq_len = input_ids.shape
input_id_shape = TensorRTShape(min_shape=[4, 1], optimal_shape=[4, 1], max_shape=[4, 200], input_name="input_ids")
encoder_hidden_states_shape = TensorRTShape(
    min_shape=[4, 1, 512],
    optimal_shape=[4, 10, 512],
    max_shape=[4, 200, 512],
    input_name="encoder_hidden_states",
)

final_seq_len = TensorRTShape(
    min_shape=[1],
    optimal_shape=[1],
    max_shape=[1],
    input_name="final_seq_len",
)

shape_tensors = [final_seq_len]
input_shapes = [input_id_shape, encoder_hidden_states_shape, final_seq_len]
for i in range(num_layers):
    input_shapes.append(
        TensorRTShape(
            min_shape=[4, 8, 0, 64],
            optimal_shape=[4, 8, 100, 64],
            max_shape=[4, 8, 200, 64],
            input_name=f"past_key_values.{i}.decoder.key",
        )
    )
    input_shapes.append(
        TensorRTShape(
            min_shape=[4, 8, 0, 64],
            optimal_shape=[4, 8, 100, 64],
            max_shape=[4, 8, 200, 64],
            input_name=f"past_key_values.{i}.decoder.value",
        )
    )
    input_shapes.append(
        TensorRTShape(
            min_shape=[4, 8, 0, 64],
            optimal_shape=[4, 8, 10, 64],
            max_shape=[4, 8, 200, 64],
            input_name=f"past_key_values.{i}.encoder.key",
        )
    )
    input_shapes.append(
        TensorRTShape(
            min_shape=[4, 8, 0, 64],
            optimal_shape=[4, 8, 10, 64],
            max_shape=[4, 8, 200, 64],
            input_name=f"past_key_values.{i}.encoder.value",
        )
    )

command_line_min = []
command_line_opt = []
command_line_max = []
for i in input_shapes:
    command_line_min.append(f"{i.input_name}:{'x'.join([str(s) for s in i.min_shape])}")
    command_line_opt.append(f"{i.input_name}:{'x'.join([str(s) for s in i.optimal_shape])}")
    command_line_max.append(f"{i.input_name}:{'x'.join([str(s) for s in i.max_shape])}")

print(
    "/usr/src/tensorrt/bin/trtexec --onnx=test-dec-if.onnx --useSpinWait --verbose --dumpLayerInfo "
    "--profilingVerbosity=detailed  --minShapes="
    + ",".join(command_line_min)
    + "  --optShapes="
    + ",".join(command_line_opt)
    + "  --maxShapes="
    + ",".join(command_line_max)
    + f"--saveEngine='{trt_model_name}' |& > logs.txt"
)


engine: ICudaEngine = build_engine(
    runtime=runtime,
    onnx_file_path="test-dec-if.onnx",
    logger=trt_logger,
    workspace_size=20000 * 1024**2,
    fp16=False,  # for tests only
    int8=False,
    input_shapes=input_shapes,
    shape_tensors=shape_tensors,
    # fp16_fix=get_fix_fp16_network_func(keep_fp32=keep_fp32),
)


save_engine(engine, trt_model_name)


tensorrt_model = load_engine(runtime=runtime, engine_file_path=trt_model_name)


c = {
    "input_ids": torch.ones((4, 1), dtype=torch.int32, device="cuda"),
    "encoder_hidden_states": torch.ones((4, 10, 512), dtype=torch.float32, device="cuda"),
    "final_seq_len": torch.tensor([1], dtype=torch.int32, device="cuda"),
    "enable_cache": torch.tensor([True], dtype=torch.bool, device="cuda"),
}

for i in range(num_layers):
    c[f"past_key_values.{i}.decoder.key"] = torch.zeros([4, 8, 100, 64], dtype=torch.float32)
    c[f"past_key_values.{i}.decoder.value"] = torch.zeros([4, 8, 100, 64], dtype=torch.float32)
    c[f"past_key_values.{i}.encoder.key"] = torch.zeros([4, 8, 10, 64], dtype=torch.float32)
    c[f"past_key_values.{i}.encoder.value"] = torch.zeros([4, 8, 10, 64], dtype=torch.float32)

for _ in range(100):
    _ = tensorrt_model(c)
start = time()
for _ in range(100):
    _ = tensorrt_model(c)
print((time() - start) / 100)
a = tensorrt_model(c)
print(a)
