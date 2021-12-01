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
from typing import List

import numpy as np
import pycuda.autoinit
import tensorrt as trt
import torch
from pycuda._driver import Stream
from tensorrt.tensorrt import IExecutionContext, Logger, Runtime
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
from transformer_deploy.benchmarks.utils import prepare_input, print_timings, setup_logging, track_infer_time
from transformer_deploy.templates.triton import Configuration, ModelType
from pytorch_quantization.nn import TensorQuantizer


def main():

    parser = argparse.ArgumentParser(
        description="optimize and deploy transformers", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model", required=True, help="path to model or URL to Hugging Face Hub")
    parser.add_argument("-t", "--tokenizer", help="path to tokenizer or URL to Hugging Face Hub")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=[1, 1, 1],
        help="batch sizes to optimize for (min, optimal, max)",
        type=int,
        nargs=3,
    )
    parser.add_argument(
        "-s",
        "--seq-len",
        default=[16, 16, 16],
        help="sequence lengths to optimize for (min, opt, max)",
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
        choices=["onnx", "tensorrt", "pytorch"],
    )
    parser.add_argument("--nb-instances", default=1, help="# of model instances, may improve troughput", type=int)
    parser.add_argument("--warmup", default=100, help="# of inferences to warm each model", type=int)
    parser.add_argument("--nb-measures", default=1000, help="# of inferences for benchmarks", type=int)
    parser.add_argument("--seed", default=123, help="seed for random inputs, etc.", type=int)
    parser.add_argument("--atol", default=1e-1, help="tolerance when comparing outputs to Pytorch ones", type=float)
    args, _ = parser.parse_known_args()

    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)

    torch.manual_seed(args.seed)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    onnx_model_path = os.path.join(args.output, "model-original.onnx")
    onnx_optim_fp16_path = os.path.join(args.output, "model.onnx")
    tensorrt_path = os.path.join(args.output, "model.plan")

    assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation"
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    input_names: List[str] = tokenizer.model_input_names
    logging.info(f"axis: {input_names}")
    include_token_ids = "token_type_ids" in input_names

    model_pytorch: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(args.model)
    model_pytorch.cuda()
    model_pytorch.eval()

    tensor_shapes = list(zip(args.batch_size, args.seq_len))
    inputs_pytorch, inputs_onnx = prepare_input(
        batch_size=tensor_shapes[-1][0], seq_len=tensor_shapes[-1][1], include_token_ids=include_token_ids
    )

    with torch.inference_mode():
        output = model_pytorch(**inputs_pytorch)
        output = output.logits  # extract the value of interest
        output_pytorch: np.ndarray = output.detach().cpu().numpy()

    logging.info(f"[Pytorch] input shape {inputs_pytorch['input_ids'].shape}")
    logging.info(f"[Pytorch] output shape: {output_pytorch.shape}")
    # create onnx model and compare results
    if args.quantization:
        TensorQuantizer.use_fb_fake_quant = True
    convert_to_onnx(model_pytorch=model_pytorch, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch)
    if args.quantization:
        TensorQuantizer.use_fb_fake_quant = False
    onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use="CUDAExecutionProvider")
    output_onnx = onnx_model.run(None, inputs_onnx)
    assert np.allclose(a=output_onnx, b=output_pytorch, atol=args.atol)
    del onnx_model

    timings = {}

    if "pytorch" in args.backend:
        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = model_pytorch(**inputs_pytorch)
                torch.cuda.synchronize()
            time_buffer = []
            for _ in range(args.nb_measures):
                with track_infer_time(time_buffer):
                    _ = model_pytorch(**inputs_pytorch)
                    torch.cuda.synchronize()
            timings["Pytorch (FP32)"] = time_buffer
            with autocast():
                for _ in range(args.warmup):
                    _ = model_pytorch(**inputs_pytorch)
                    torch.cuda.synchronize()
                time_buffer = []
                for _ in range(args.nb_measures):
                    with track_infer_time(time_buffer):
                        _ = model_pytorch(**inputs_pytorch)
                        torch.cuda.synchronize()
                timings["Pytorch (FP16)"] = time_buffer
    del model_pytorch

    if "tensorrt" in args.backend:
        trt_logger: Logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
        runtime: Runtime = trt.Runtime(trt_logger)
        engine = build_engine(
            runtime=runtime,
            onnx_file_path=onnx_model_path,
            logger=trt_logger,
            min_shape=tensor_shapes[0],
            optimal_shape=tensor_shapes[1],
            max_shape=tensor_shapes[2],
            workspace_size=args.workspace_size * 1024 * 1024,
            fp16=True,
            int8=args.quantization,
        )
        save_engine(engine=engine, engine_file_path=tensorrt_path)
        # important to check the engine has been correctly serialized
        engine = load_engine(runtime=runtime, engine_file_path=tensorrt_path)
        stream: Stream = pycuda.driver.Stream()
        context: IExecutionContext = engine.create_execution_context()
        profile_index = 0
        context.set_optimization_profile_async(profile_index=profile_index, stream_handle=stream.handle)
        # retrieve input/output IDs
        input_binding_idxs, output_binding_idxs = get_binding_idxs(engine, profile_index)  # type: List[int], List[int]
        tensorrt_output = infer_tensorrt(
            context=context,
            host_inputs=inputs_onnx,
            input_binding_idxs=input_binding_idxs,
            output_binding_idxs=output_binding_idxs,
            stream=stream,
        )
        assert np.allclose(a=tensorrt_output, b=output_pytorch, atol=args.atol), (
            f"tensorrt accuracy is too low:\n" f"Pythorch:\n{output_pytorch}\n" f"VS\n" f"TensorRT:\n{tensorrt_output}"
        )

        for _ in range(args.warmup):
            _ = infer_tensorrt(
                context=context,
                host_inputs=inputs_onnx,
                input_binding_idxs=input_binding_idxs,
                output_binding_idxs=output_binding_idxs,
                stream=stream,
            )
        time_buffer = list()
        for _ in range(args.nb_measures):
            with track_infer_time(time_buffer):
                _ = infer_tensorrt(
                    context=context,
                    host_inputs=inputs_onnx,
                    input_binding_idxs=input_binding_idxs,
                    output_binding_idxs=output_binding_idxs,
                    stream=stream,
                )
            timings["TensorRT (FP16)"] = time_buffer
        del engine, context, runtime  # delete all tensorrt objects

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
        onnx_model = create_model_for_provider(path=onnx_optim_fp16_path, provider_to_use="CUDAExecutionProvider")
        # run the model (None = get all the outputs)
        output_onnx_optimised = onnx_model.run(None, inputs_onnx)
        del onnx_model
        assert np.allclose(a=output_onnx_optimised, b=output_pytorch, atol=args.atol)

        for provider, model_path, benchmar_name in [
            ("CUDAExecutionProvider", onnx_model_path, "ONNX Runtime (vanilla)"),
            ("CUDAExecutionProvider", onnx_optim_fp16_path, "ONNX Runtime (optimized)"),
        ]:
            model = create_model_for_provider(path=model_path, provider_to_use=provider)
            time_buffer = []
            for _ in range(args.warmup):
                _ = model.run(None, inputs_onnx)
            for _ in range(args.nb_measures):
                with track_infer_time(time_buffer):
                    _ = model.run(None, inputs_onnx)
            timings[benchmar_name] = time_buffer
        del model

    print(f"Inference done on {get_device_name(0)}")
    print("latencies:")
    for name, time_buffer in timings.items():
        print_timings(name=name, timings=time_buffer)

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


if __name__ == "__main__":
    main()
