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
import pycuda.autoinit
import tensorrt as trt
import torch
from numpy import ndarray
from torch.cuda import get_device_name
from torch.cuda.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from transformer_deploy.backends.ort_utils import convert_to_onnx, create_model_for_provider, optimize_onnx
from transformer_deploy.backends.trt_utils import (
    build_engine,
    get_binding_idxs,
    infer_tensorrt,
    load_engine,
    save_engine,
)
from transformer_deploy.benchmarks.utils import (
    compare_outputs,
    generate_multiple_inputs,
    print_timings,
    setup_logging,
    track_infer_time,
)
from transformer_deploy.templates.triton import Configuration, ModelType


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


def main():
    parser = argparse.ArgumentParser(
        description="optimize and deploy transformers", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model", required=True, help="path to model or URL to Hugging Face hub")
    parser.add_argument("-t", "--tokenizer", help="path to tokenizer or URL to Hugging Face hub")
    parser.add_argument(
        "--auth-token",
        default=None,
        help=(
            "Hugging Face Hub auth token. Set to `None` (default) for public models. "
            "For private models, use `True` to use local cached token, or a string of your HF API token"
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=[1, 1, 1],
        help="batch sizes to optimize for (min, optimal, max). For TensorRT and benchmarks.",
        type=int,
        nargs=3,
    )
    parser.add_argument(
        "-s",
        "--seq-len",
        default=[16, 16, 16],
        help="sequence lengths to optimize for (min, opt, max). For TensorRT and benchmarks.",
        type=int,
        nargs=3,
    )
    parser.add_argument("-q", "--quantization", action="store_true", help="int-8 GPU quantization support")
    parser.add_argument("-w", "--workspace-size", default=10000, help="workspace size in MiB (TensorRT)", type=int)
    parser.add_argument("-o", "--output", default="triton_models", help="name to be used for ")
    parser.add_argument("-n", "--name", default="transformer", help="model name to be used in triton server")
    parser.add_argument("-v", "--verbose", action="store_true", help="display detailed information")
    parser.add_argument(
        "--backend",
        default=["onnx"],
        help="backend to use. One of [onnx,tensorrt, pytorch] or all",
        nargs="*",
        choices=["onnx", "tensorrt"],
    )
    parser.add_argument("--nb-instances", default=1, help="# of model instances, may improve troughput", type=int)
    parser.add_argument("--warmup", default=10, help="# of inferences to warm each model", type=int)
    parser.add_argument("--nb-measures", default=1000, help="# of inferences for benchmarks", type=int)
    parser.add_argument("--seed", default=123, help="seed for random inputs, etc.", type=int)
    parser.add_argument("--atol", default=3e-1, help="tolerance when comparing outputs to Pytorch ones", type=float)
    args, _ = parser.parse_known_args()

    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)

    if len(args.seq_len) == len(set(args.seq_len)) and "tensorrt" in args.backend:
        logging.warning("having different sequence lengths may make TensorRT slower")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if isinstance(args.auth_token, str) and args.auth_token.lower() in ["true", "t"]:
        auth_token = True
    elif isinstance(args.auth_token, str):
        auth_token = args.auth_token
    else:
        auth_token = None

    Path(args.output).mkdir(parents=True, exist_ok=True)
    onnx_model_path = os.path.join(args.output, "model-original.onnx")
    onnx_optim_fp16_path = os.path.join(args.output, "model.onnx")
    tensorrt_path = os.path.join(args.output, "model.plan")

    assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation"
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=auth_token)
    input_names: List[str] = tokenizer.model_input_names
    logging.info(f"axis: {input_names}")
    include_token_ids = "token_type_ids" in input_names
    model_pytorch: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.model, use_auth_token=auth_token
    )
    model_pytorch.cuda()
    model_pytorch.eval()

    tensor_shapes = list(zip(args.batch_size, args.seq_len))
    # take optimial size
    inputs_pytorch, inputs_onnx = generate_multiple_inputs(
        batch_size=tensor_shapes[1][0],
        seq_len=tensor_shapes[1][1],
        include_token_ids=include_token_ids,
        device="cuda",
        nb_inputs_to_gen=args.warmup,
    )
    input_pytorch = inputs_pytorch[0]
    input_onnx = inputs_onnx[0]

    with torch.inference_mode():
        output = model_pytorch(**input_pytorch)
        output_pytorch: np.ndarray = output.logits.detach().cpu().numpy()

    logging.info(f"[Pytorch] input shape {input_pytorch['input_ids'].shape}")
    logging.info(f"[Pytorch] output shape: {output_pytorch.shape}")
    # create onnx model and compare results
    opset = 12
    if args.quantization:
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

    convert_to_onnx(model_pytorch=model_pytorch, output_path=onnx_model_path, inputs_pytorch=input_pytorch, opset=opset)
    if args.quantization:
        TensorQuantizer.use_fb_fake_quant = False
    onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use="CUDAExecutionProvider")
    output_onnx = onnx_model.run(None, input_onnx)
    assert np.allclose(a=output_onnx, b=output_pytorch, atol=args.atol)
    del onnx_model

    timings = {}

    def infer_classification_pytorch(inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        output = model_pytorch(**inputs)
        output_formated = output.logits.detach().cpu().numpy()
        torch.cuda.synchronize()
        return output_formated

    with torch.inference_mode():
        pytorch_output, time_buffer = launch_inference(
            infer=infer_classification_pytorch, inputs=inputs_pytorch, nb_measures=args.nb_measures
        )
        timings["Pytorch (FP32)"] = time_buffer
        with autocast():
            engine_name = "Pytorch (FP16)"
            pytorch_fp16_output, time_buffer = launch_inference(
                infer=infer_classification_pytorch, inputs=inputs_pytorch, nb_measures=args.nb_measures
            )
            check_accuracy(
                engine_name=engine_name,
                pytorch_output=pytorch_output,
                engine_output=pytorch_fp16_output,
                tolerance=args.atol,
            )
            timings[engine_name] = time_buffer
    del model_pytorch

    if "tensorrt" in args.backend:
        try:
            from pycuda._driver import Stream
            from tensorrt.tensorrt import ICudaEngine, IExecutionContext, Logger, Runtime
        except ImportError:
            raise ImportError(
                "It seems that pycuda and TensorRT are not yet installed. "
                "They are required when you declare TensorRT backend."
                "Please find installation instruction on "
                "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
            )

        trt_logger: Logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
        runtime: Runtime = trt.Runtime(trt_logger)
        engine: ICudaEngine = build_engine(
            runtime=runtime,
            onnx_file_path=onnx_model_path,
            logger=trt_logger,
            min_shape=tensor_shapes[0],
            optimal_shape=tensor_shapes[1],
            max_shape=tensor_shapes[2],
            workspace_size=args.workspace_size * 1024 * 1024,
            fp16=not args.quantization,
            int8=args.quantization,
        )
        save_engine(engine=engine, engine_file_path=tensorrt_path)
        # important to check the engine has been correctly serialized
        tensorrt_model: Callable[[Dict[str, ndarray]], ndarray] = load_engine(
            runtime=runtime, engine_file_path=tensorrt_path
        )

        engine_name = "TensorRT (FP16)"
        tensorrt_output, time_buffer = launch_inference(
            infer=tensorrt_model, inputs=inputs_onnx, nb_measures=args.nb_measures
        )
        check_accuracy(
            engine_name=engine_name,
            pytorch_output=pytorch_output,
            engine_output=tensorrt_output,
            tolerance=args.atol,
        )
        timings[engine_name] = time_buffer
        del engine, tensorrt_model, runtime  # delete all tensorrt objects
        conf = Configuration(
            model_name=args.name,
            model_type=ModelType.TensorRT,
            batch_size=0,
            nb_output=output_pytorch.shape[1],
            nb_instance=args.nb_instances,
            include_token_type=include_token_ids,
            workind_directory=args.output,
        )
        conf.create_folders(tokenizer=tokenizer, model_path=tensorrt_path)

    if "onnx" in args.backend:
        # create optimized onnx model and compare results
        optimize_onnx(
            onnx_path=onnx_model_path,
            onnx_optim_fp16_path=onnx_optim_fp16_path,
            use_cuda=True,
        )

        for provider, model_path, benchmark_name in [
            ("CUDAExecutionProvider", onnx_model_path, "ONNX Runtime (FP32)"),
            ("CUDAExecutionProvider", onnx_optim_fp16_path, "ONNX Runtime (FP16)"),
        ]:
            ort_model = create_model_for_provider(path=model_path, provider_to_use=provider)

            def infer_ort(inputs: Dict[str, np.ndarray]) -> np.ndarray:
                return ort_model.run(None, inputs)

            ort_output, time_buffer = launch_inference(
                infer=infer_ort, inputs=inputs_onnx, nb_measures=args.nb_measures
            )
            check_accuracy(
                engine_name=benchmark_name,
                pytorch_output=pytorch_output,
                engine_output=ort_output,
                tolerance=args.atol,
            )
            timings[benchmark_name] = time_buffer
            del ort_model

        conf = Configuration(
            model_name=args.name,
            model_type=ModelType.ONNX,
            batch_size=0,
            nb_output=output_pytorch.shape[1],
            nb_instance=args.nb_instances,
            include_token_type=include_token_ids,
            workind_directory=args.output,
        )
        conf.create_folders(tokenizer=tokenizer, model_path=onnx_optim_fp16_path)

    print(f"Inference done on {get_device_name(0)}")
    print("latencies:")
    for name, time_buffer in timings.items():
        print_timings(name=name, timings=time_buffer)


if __name__ == "__main__":
    main()
