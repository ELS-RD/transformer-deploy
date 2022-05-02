from typing import Dict, List, Set

import onnx
import torch
from onnx import ModelProto
from transformers import AutoTokenizer, T5TokenizerFast

from transformer_deploy.backends.ort_utils import (
    add_output_nodes,
    create_model_for_provider,
    get_adjency_dict,
    get_list_fp32_nodes,
    inference_onnx_binding,
)

model_name = "t5-small"
tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(model_name)

onnx_model: ModelProto = onnx.load("test-dec-no-cache.onnx")
onnx_all_nodes = add_output_nodes(model=onnx_model)
onnx_graph: Dict[str, Set[str]] = get_adjency_dict(model=onnx_model)
ort_model = create_model_for_provider(onnx_all_nodes.SerializeToString(), "CUDAExecutionProvider")
enc_ort_model = create_model_for_provider("test-enc.onnx", "CUDAExecutionProvider")


# TODO apply the process to the 2 subgraphs directly but not the graph whole if graph
# use info from tokenizer size and max shape provided through the command line
def get_random_input() -> Dict[str, torch.Tensor]:
    inputs = {"input_ids": torch.randint(low=0, high=tokenizer.vocab_size, size=(4, 1000), dtype=torch.int32, device="cuda")}
    encoder_hidden_states = inference_onnx_binding(
        model_onnx=enc_ort_model,
        inputs=inputs,
        device="cuda",
    )["output"]
    inputs["encoder_hidden_states"] = encoder_hidden_states
    return inputs


keep_fp32 = get_list_fp32_nodes(onnx_graph=onnx_graph, model=ort_model, get_input=get_random_input, nb_try=200)
print(keep_fp32)

# thread sur FP16 marche pas
# https://github.com/microsoft/onnxruntime/issues/11119
