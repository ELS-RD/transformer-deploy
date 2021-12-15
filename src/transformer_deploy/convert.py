#!/usr/bin/env python3

#  Copyright 2021, Lefebvre Sarrut Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from numpy import ndarray
from torch.cuda import get_device_name
from torch.cuda.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from transformer_deploy.backends.ort_utils import convert_to_onnx, create_model_for_provider, optimize_onnx
from transformer_deploy.backends.trt_utils import build_engine, load_engine, save_engine
from transformer_deploy.benchmarks.utils import (
    compare_outputs,
    generate_multiple_inputs,
    print_timings,
    setup_logging,
    track_infer_time,
)
from transformer_deploy.templates.triton import Configuration, ModelType
from transformer_deploy.utils.args import parse_args


def check_accuracy(
    engine_name: str, pytorch_output: List[np.ndarray], engine_output: List[np.ndarray], tolerance: float
):
    discrepency = compare_outputs(pytorch_output=pytorch_output, engine_output=engine_output)
    assert discrepency < tolerance, (
        f"{engine_name} discrepency is too high ({discrepency:.2f} > {tolerance}):\n"
        f"Pythorch:\n{pytorch_output}\n"
        f"VS\n"
        f"{engine_name}:\n{engine_output}\n"
        f"Diff:\n"
        f"{np.asarray(pytorch_output) - np.asarray(engine_output)}"
    )


def launch_inference(
    infer: Callable, inputs: List[Dict[str, Union[np.ndarray, torch.Tensor]]], nb_measures: int
) -> Tuple[List[np.ndarray], List[float]]:
    assert type(inputs) == list
    assert len(inputs) > 0
    outputs = list()
    for batch_input in inputs:
        output = infer(batch_input)
        outputs.append(output)
    time_buffer: List[float] = list()
    for _ in range(nb_measures):
        with track_infer_time(time_buffer):
            _ = infer(inputs[0])
    return outputs, time_buffer


def main(commands: argparse.Namespace):
    setup_logging(level=logging.INFO if commands.verbose else logging.WARNING)
    if commands.device == "cpu" and "tensorrt" in commands.backend:
        raise Exception("can't perform inference on CPU and use Nvidia TensorRT as backend")
    if len(commands.seq_len) == len(set(commands.seq_len)) and "tensorrt" in commands.backend:
        logging.warning("having different sequence lengths may make TensorRT slower")

    torch.manual_seed(commands.seed)
    np.random.seed(commands.seed)

    if isinstance(commands.auth_token, str) and commands.auth_token.lower() in ["true", "t"]:
        auth_token = True
    elif isinstance(commands.auth_token, str):
        auth_token = commands.auth_token
    else:
        auth_token = None

    Path(commands.output).mkdir(parents=True, exist_ok=True)
    onnx_model_path = os.path.join(commands.output, "model-original.onnx")
    onnx_optim_fp16_path = os.path.join(commands.output, "model.onnx")
    tensorrt_path = os.path.join(commands.output, "model.plan")
    if commands.device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation"
    tokenizer_path = commands.tokenizer if commands.tokenizer else commands.model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=auth_token)
    input_names: List[str] = tokenizer.model_input_names
    logging.info(f"axis: {input_names}")
    include_token_ids = "token_type_ids" in input_names
    model_pytorch: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        commands.model, use_auth_token=auth_token
    )
    model_pytorch.eval()
    if commands.device == "cuda":
        model_pytorch.cuda()

    tensor_shapes = list(zip(commands.batch_size, commands.seq_len))
    # take optimial size
    inputs_pytorch, inputs_onnx = generate_multiple_inputs(
        batch_size=tensor_shapes[1][0],
        seq_len=tensor_shapes[1][1],
        include_token_ids=include_token_ids,
        device=commands.device,
        nb_inputs_to_gen=commands.warmup,
    )

    # create onnx model and compare results
    opset = 12
    if commands.quantization and commands.device == "cuda":
        try:
            from pytorch_quantization.nn import TensorQuantizer
        except ImportError:
            raise ImportError(
                "It seems that pytorch-quantization is not yet installed. "
                "It is required when you enable the quantization flag."
                "Please find installation instruction on "
                "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization"
            )

        TensorQuantizer.use_fb_fake_quant = True
        opset = 13

    convert_to_onnx(
        model_pytorch=model_pytorch, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch[0], opset=opset
    )
    if commands.quantization and commands.device == "cuda":
        TensorQuantizer.use_fb_fake_quant = False

    timings = {}

    def infer_classification_pytorch(inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        model_output = model_pytorch(**inputs).logits.detach().cpu().numpy()  # noqa: F821
        if commands.device == "cuda":
            torch.cuda.synchronize()
        return model_output

    with torch.inference_mode():
        pytorch_output, time_buffer = launch_inference(
            infer=infer_classification_pytorch, inputs=inputs_pytorch, nb_measures=commands.nb_measures
        )
        timings["Pytorch (FP32)"] = time_buffer
        if commands.device == "cuda":
            with autocast():
                engine_name = "Pytorch (FP16)"
                pytorch_fp16_output, time_buffer = launch_inference(
                    infer=infer_classification_pytorch, inputs=inputs_pytorch, nb_measures=commands.nb_measures
                )
                check_accuracy(
                    engine_name=engine_name,
                    pytorch_output=pytorch_output,
                    engine_output=pytorch_fp16_output,
                    tolerance=commands.atol,
                )
                timings[engine_name] = time_buffer
    del model_pytorch

    if "tensorrt" in commands.backend:
        try:
            from tensorrt.tensorrt import ICudaEngine, Logger, Runtime
        except ImportError:
            raise ImportError(
                "It seems that pycuda and TensorRT are not yet installed. "
                "They are required when you declare TensorRT backend."
                "Please find installation instruction on "
                "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
            )

        trt_logger: Logger = trt.Logger(trt.Logger.INFO if commands.verbose else trt.Logger.WARNING)
        runtime: Runtime = trt.Runtime(trt_logger)
        engine: ICudaEngine = build_engine(
            runtime=runtime,
            onnx_file_path=onnx_model_path,
            logger=trt_logger,
            min_shape=tensor_shapes[0],
            optimal_shape=tensor_shapes[1],
            max_shape=tensor_shapes[2],
            workspace_size=commands.workspace_size * 1024 * 1024,
            fp16=not commands.quantization,
            int8=commands.quantization,
        )
        save_engine(engine=engine, engine_file_path=tensorrt_path)
        # important to check the engine has been correctly serialized
        tensorrt_model: Callable[[Dict[str, ndarray]], ndarray] = load_engine(
            runtime=runtime, engine_file_path=tensorrt_path
        )

        engine_name = "TensorRT (FP16)"
        tensorrt_output, time_buffer = launch_inference(
            infer=tensorrt_model, inputs=inputs_onnx, nb_measures=commands.nb_measures
        )
        check_accuracy(
            engine_name=engine_name,
            pytorch_output=pytorch_output,
            engine_output=tensorrt_output,
            tolerance=commands.atol,
        )
        timings[engine_name] = time_buffer
        del engine, tensorrt_model, runtime  # delete all tensorrt objects
        conf = Configuration(
            model_name=commands.name,
            model_type=ModelType.TensorRT,
            batch_size=0,
            nb_output=pytorch_output[0].shape[1],
            nb_instance=commands.nb_instances,
            include_token_type=include_token_ids,
            workind_directory=commands.output,
        )
        conf.create_folders(tokenizer=tokenizer, model_path=tensorrt_path)

    if "onnx" in commands.backend:
        # create optimized onnx model and compare results
        optimize_onnx(
            onnx_path=onnx_model_path,
            onnx_optim_fp16_path=onnx_optim_fp16_path,
            use_cuda=commands.device == "cuda",
        )
        ort_provider = "CUDAExecutionProvider" if commands.device == "cuda" else "CPUExecutionProvider"
        for provider, model_path, benchmark_name in [
            (ort_provider, onnx_model_path, "ONNX Runtime (FP32)"),
            (ort_provider, onnx_optim_fp16_path, "ONNX Runtime (FP16)"),
        ]:
            ort_model = create_model_for_provider(path=model_path, provider_to_use=provider)

            def infer_ort(inputs: Dict[str, np.ndarray]) -> np.ndarray:
                return ort_model.run(None, inputs)

            ort_output, time_buffer = launch_inference(
                infer=infer_ort, inputs=inputs_onnx, nb_measures=commands.nb_measures
            )
            check_accuracy(
                engine_name=benchmark_name,
                pytorch_output=pytorch_output,
                engine_output=ort_output,
                tolerance=commands.atol,
            )
            timings[benchmark_name] = time_buffer
            del ort_model

        conf = Configuration(
            model_name=commands.name,
            model_type=ModelType.ONNX,
            batch_size=0,
            nb_output=pytorch_output[0].shape[1],
            nb_instance=commands.nb_instances,
            include_token_type=include_token_ids,
            workind_directory=commands.output,
        )
        conf.create_folders(tokenizer=tokenizer, model_path=onnx_optim_fp16_path)

    if commands.device == "cuda":
        print(f"Inference done on {get_device_name(0)}")

    print("latencies:")
    for name, time_buffer in timings.items():
        print_timings(name=name, timings=time_buffer)


def entrypoint():
    args = parse_args()
    main(commands=args)


if __name__ == "__main__":
    entrypoint()
