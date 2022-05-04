from time import time
from typing import Dict

import matplotlib.pylab as plt
import numpy as np
import onnx
import torch
from onnx import ModelProto
from transformers import AutoTokenizer, T5TokenizerFast

from transformer_deploy.backends.ort_utils import (
    add_output_nodes,
    convert_fp16,
    create_model_for_provider,
    get_list_fp32_nodes,
    inference_onnx_binding,
)


model_name = "t5-base"
tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(model_name)
enc_ort_model = create_model_for_provider("test-enc.onnx", "CUDAExecutionProvider")
dec_no_cache_ort_model = create_model_for_provider("test-dec-no-cache.onnx", "CUDAExecutionProvider")

onnx_model: ModelProto = onnx.load("test-dec-cache.onnx")
onnx_all_nodes = add_output_nodes(model=onnx_model)
ort_model = create_model_for_provider(onnx_all_nodes.SerializeToString(), "CUDAExecutionProvider")


# thread sur FP16 marche pas
# https://github.com/microsoft/onnxruntime/issues/11119=
# use info from tokenizer size and max shape provided through the command line
def get_random_input() -> Dict[str, torch.Tensor]:
    batch = 4
    seq_len = 512
    input_ids = torch.randint(low=2, high=tokenizer.vocab_size, size=(batch, seq_len), dtype=torch.int32, device="cuda")
    inputs = {"input_ids": input_ids}
    encoder_hidden_states = inference_onnx_binding(
        model_onnx=enc_ort_model,
        inputs=inputs,
        device="cuda",
        clone_tensor=False,
    )["output"]
    inputs["encoder_hidden_states"] = encoder_hidden_states
    dec_past_states = inference_onnx_binding(
        model_onnx=dec_no_cache_ort_model,
        inputs=inputs,
        device="cuda",
        clone_tensor=False,
    )
    for k, v in dec_past_states.items():
        if k == "logits":
            continue
        new_k = k.replace("present", "past_key_values")
        inputs[new_k] = v
    complement = torch.randint(low=0, high=tokenizer.vocab_size, size=(batch, 1), dtype=torch.int32, device="cuda")
    inputs["input_ids"] = torch.concat(tensors=[input_ids, complement], dim=1)
    return inputs


keep_fp32 = get_list_fp32_nodes(onnx_model=onnx_model, ort_model=ort_model, get_input=get_random_input, nb_try=1000)

model_fp16 = onnx.load("test-dec-cache.onnx")
model_fp16 = convert_fp16(onnx_model=model_fp16, nodes_to_exclude=keep_fp32)
dec_cache_fp16_ort_model = create_model_for_provider(model_fp16.SerializeToString(), "CUDAExecutionProvider")

dec_cache_fp_32_ort_model = create_model_for_provider("test-dec-cache.onnx", "CUDAExecutionProvider")

fp_16_timings = list()
fp_32_timings = list()
nb_try = 100
for _ in range(nb_try):
    random_input = get_random_input()
    torch.cuda.synchronize()
    start = time()
    res_fp16 = inference_onnx_binding(
        model_onnx=dec_cache_fp16_ort_model,
        inputs=random_input,
        device="cuda",
        clone_tensor=False,
    )["logits"]
    fp_16_timings.append(time() - start)
    start = time()
    res_fp32 = inference_onnx_binding(
        model_onnx=dec_cache_fp_32_ort_model,
        inputs=random_input,
        device="cuda",
        clone_tensor=False,
    )["logits"]
    fp_32_timings.append(time() - start)
    assert np.allclose(a=res_fp32.detach().cpu().numpy(), b=res_fp16.detach().cpu().numpy(), atol=5e-1)


axis = range(nb_try)
plt.scatter(axis, fp_32_timings, marker="o", color="red", label="fp32", s=1)
plt.scatter(axis, fp_16_timings, marker="o", color="purple", label="fp16", s=1)
plt.axhline(y=np.mean(fp_32_timings), color="red", linestyle="-")
plt.axhline(y=np.mean(fp_16_timings), color="purple", linestyle="-")
plt.legend()
plt.show()
