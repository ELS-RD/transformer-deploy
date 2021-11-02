import logging
import multiprocessing
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from torch.cuda import get_device_name
from torch.cuda.amp import autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TensorType,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy

from utils import print_timings, setup_logging, track_infer_time

onnx_folder = "./onnx_models"
Path(onnx_folder).mkdir(parents=True, exist_ok=True)
onnx_model_path = os.path.join(onnx_folder, "model.onnx")
infered_shape_model_onnx_path = os.path.join(onnx_folder, "model-shape.onnx")
onnx_optim_fp16_path_path = os.path.join(onnx_folder, "model-optimized.onnx")
huggingface_hub_path = "philschmid/MiniLM-L6-H384-uncased-sst2"
input_text = "This live event is great. I will sign-up for Infinity."


def create_model_for_provider(path: str, provider_to_use: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if type(provider_to_use) == list and provider_to_use == "CPUExecutionProvider":
        options.intra_op_num_threads = multiprocessing.cpu_count()
    if type(provider_to_use) != list:
        provider_to_use = [provider_to_use]
    return InferenceSession(path, options, providers=provider_to_use)


def tokenize(dummy_text: List[str], tokenizer: PreTrainedTokenizer, type: TensorType) -> BatchEncoding:
    return tokenizer(
        text=dummy_text,
        add_special_tokens=True,
        max_length=16,
        padding=PaddingStrategy.LONGEST,
        pad_to_multiple_of=8,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_tensors=type,
        truncation=TruncationStrategy.LONGEST_FIRST,
    )


def prepare_input(
    dummy_text: str, tokenizer: PreTrainedTokenizer
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    inputs_pytorch = tokenize(dummy_text=[dummy_text] * 1, tokenizer=tokenizer, type=TensorType.PYTORCH)
    inputs_pytorch = {k: v.to(device="cuda") for k, v in inputs_pytorch.items()}
    inputs_onnx: Dict[str, np.ndarray] = {
        k: np.ascontiguousarray(v.detach().cpu().numpy()) for k, v in inputs_pytorch.items()
    }
    return inputs_pytorch, inputs_onnx


def convert_to_onnx(model_pytorch: PreTrainedModel, output_path: str, inputs_pytorch: Dict[str, torch.Tensor]) -> None:
    symbolic_names = {0: "batch_size", 1: "sequence"}

    with torch.no_grad():
        torch.onnx.export(
            model_pytorch,  # model being run
            args=tuple(inputs_pytorch.values()),  # model input (or a tuple for multiple inputs)
            f=output_path,
            # where to save the model (can be a file or file-like object)
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input_ids", "attention_mask"],  # the model's input names
            output_names=["model_output"],  # the model's output names
            dynamic_axes={
                "input_ids": symbolic_names,  # variable length axes
                "attention_mask": symbolic_names,
                "model_output": {0: "batch_size"},
            },
            verbose=False,
        )


def optimize_onnx(onnx_path: str, onnx_optim_fp16_path: str, use_cuda: bool) -> None:
    optimization_options = FusionOptions("bert")
    # optimization_options.enable_gelu_approximation = True
    optimized_model: BertOnnxModel = optimizer.optimize_model(
        input=onnx_path,
        model_type="bert",
        use_gpu=use_cuda,
        opt_level=1,
        num_heads=0,
        hidden_size=0,
        optimization_options=optimization_options,
    )

    optimized_model.convert_float_to_float16()
    logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(onnx_optim_fp16_path)


setup_logging()
model_pytorch: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(huggingface_hub_path)
assert torch.cuda.is_available()
model_pytorch.cuda()
model_pytorch.eval()

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(huggingface_hub_path)

inputs_pytorch, inputs_onnx = prepare_input(dummy_text=input_text, tokenizer=tokenizer)

with torch.no_grad():
    output = model_pytorch(**inputs_pytorch)
    output = output.logits  # extract the value of interest
    output_pytorch: np.ndarray = output.detach().cpu().numpy()

logging.info(f"[Pytorch] input shape {inputs_pytorch['input_ids'].shape}")
logging.info(f"[Pytorch] output shape: {output_pytorch.shape}")
# create onnx model and compare results
convert_to_onnx(model_pytorch=model_pytorch, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch)
onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use="CPUExecutionProvider")
output_onnx = onnx_model.run(None, inputs_onnx)
del onnx_model
# del model_pytorch
assert np.allclose(a=output_onnx, b=output_pytorch, atol=1e-1)
# import after onnxruntime
import onnx
onnx.save(
    SymbolicShapeInference.infer_shapes(onnx.load(onnx_model_path), auto_merge=True), infered_shape_model_onnx_path
)

# create optimized onnx model and compare results
optimize_onnx(
    onnx_path=onnx_model_path,
    onnx_optim_fp16_path=onnx_optim_fp16_path_path,
    use_cuda=True,
)
onnx_model = create_model_for_provider(path=onnx_optim_fp16_path_path, provider_to_use="CPUExecutionProvider")
# run the model (None = get all the outputs)
output_onnx_optimised = onnx_model.run(None, inputs_onnx)
del onnx_model
assert np.allclose(a=output_onnx_optimised, b=output_pytorch, atol=1e-1)
assert np.allclose(a=output_onnx_optimised, b=output_onnx, atol=1e-1)

os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
os.environ["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = f"{3 * 1024 * 1024 * 1024}"
# os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'

providers = [
    ("TensorrtExecutionProvider", infered_shape_model_onnx_path),
    ("CUDAExecutionProvider", onnx_model_path),
    ("CUDAExecutionProvider", onnx_optim_fp16_path_path),
]

warm_up = 100
nb_benchmark = 1000
results = {}
for provider, model_path in providers:
    model = create_model_for_provider(path=model_path, provider_to_use=provider)
    time_buffer = []
    for _ in range(warm_up):
        model.run(None, inputs_onnx)
    for _ in range(nb_benchmark):
        with track_infer_time(time_buffer):
            model.run(None, inputs_onnx)
    results[f"[{provider}] {model_path}"] = time_buffer
del model

# Add PyTorch to the providers
for _ in range(warm_up):
    res = model_pytorch(**inputs_pytorch)
time_buffer = []
for _ in range(nb_benchmark):
    with track_infer_time(time_buffer):
        model_pytorch(**inputs_pytorch)
results["Pytorch_fp32"] = time_buffer

with autocast():
    for _ in range(warm_up):
        model_pytorch(**inputs_pytorch)
    time_buffer = []
    for _ in range(nb_benchmark):
        with track_infer_time(time_buffer):
            model_pytorch(**inputs_pytorch)
    results["Pytorch_fp16"] = time_buffer

logging.info(f"inference done on {get_device_name(0)}")

for name, time_buffer in results.items():
    print_timings(name=name, timings=time_buffer)
