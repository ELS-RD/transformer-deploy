from collections import defaultdict
from typing import Callable, Dict, Set

import numpy as np
import onnx
import tensorrt as trt
import torch
from onnx import ModelProto
from tensorrt import ICudaEngine, Logger, Runtime
from torch.nn import Linear
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast, TensorType
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding
from transformer_deploy.backends.pytorch_utils import convert_to_onnx
from transformer_deploy.backends.trt_utils import (
    add_output_nodes,
    build_engine,
    find_node_fp32,
    get_adjency_dict,
    get_all_outputs,
    get_fix_fp16_network_func,
    load_engine,
    save_engine,
)


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


model_onnx: ModelProto = onnx.load("test-enc.onnx")
model_onnx_all_nodes = add_output_nodes(model=model_onnx)
onnx_graph: Dict[str, Set[str]] = get_adjency_dict(model=model_onnx)
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
