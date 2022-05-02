from typing import Dict, Set

import onnx
import torch
from onnx import ModelProto
from transformers import AutoTokenizer, T5TokenizerFast

from transformer_deploy.backends.ort_utils import (
    add_output_nodes,
    create_model_for_provider,
    get_adjency_dict,
    get_list_fp32_nodes,
)

model_name = "t5-small"
tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(model_name)

enc_onnx = create_model_for_provider("test-enc.onnx", "CUDAExecutionProvider")

# https://github.com/microsoft/onnxruntime/issues/1455#issuecomment-979901463
# add all intermediate outputs to onnx net
# org_outputs = [x.name for x in enc_onnx.get_outputs()]


model_onnx: ModelProto = onnx.load("test-enc.onnx")
model_onnx_all_nodes = add_output_nodes(model=model_onnx)
onnx_graph: Dict[str, Set[str]] = get_adjency_dict(model=model_onnx)
ort_model_all_nodes = create_model_for_provider(model_onnx_all_nodes.SerializeToString(), "CUDAExecutionProvider")


# use info from tokenizer size and max shape provided through the command line
def get_random_input() -> Dict[str, torch.Tensor]:
    return {"input_ids": torch.randint(low=0, high=32100, size=(4, 2000), dtype=torch.int32, device="cuda")}


keep_fp32 = get_list_fp32_nodes(onnx_graph=onnx_graph, model=ort_model_all_nodes, get_input=get_random_input, nb_try=2)
print(keep_fp32)
