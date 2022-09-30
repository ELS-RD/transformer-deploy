#!/usr/bin/env python3

#  Copyright 2022, Lefebvre Dalloz Services
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

"""
This module contains code related to client interface.
"""

import argparse
import gc
import logging
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer, AutoModelForSeq2SeqLM, TensorType,
)

from transformer_deploy.backends.ort_utils import (
    optimize_onnx,
)
from transformer_deploy.backends.pytorch_utils import (
    get_model_size,
    infer_classification_pytorch,
)
from transformer_deploy.benchmarks.utils import print_timings, setup_logging
from transformer_deploy.convert import launch_inference
from transformer_deploy.triton.configuration import EngineType
from transformer_deploy.utils.args import parse_args
from transformer_deploy.utils.t5_inference_utils import ExtT5
from transformer_deploy.utils.t5_utils import create_triton_configs, convert_t5_to_onnx


def decode_model_outputs(tokenizer: PreTrainedTokenizer, model_output, benchmark_name: str):
    print(
        f"{benchmark_name}: "
        f"{tokenizer.decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)}"
    )


def main(commands: argparse.Namespace):
    setup_logging(level=logging.INFO if commands.verbose else logging.WARNING)
    logging.info("running with commands: %s", commands)

    if commands.device is None:
        commands.device = "cuda" if torch.cuda.is_available() else "cpu"

    if commands.device == "cpu" and "tensorrt" in commands.backend:
        raise Exception("can't perform inference on CPU and use Nvidia TensorRT as backend")

    if len(commands.seq_len) == len(set(commands.seq_len)) and "tensorrt" in commands.backend:
        logging.warning("having different sequence lengths may make TensorRT slower")

    torch.manual_seed(commands.seed)
    np.random.seed(commands.seed)
    torch.set_num_threads(commands.nb_threads)

    if isinstance(commands.auth_token, str) and commands.auth_token.lower() in ["true", "t"]:
        auth_token = True
    elif isinstance(commands.auth_token, str):
        auth_token = commands.auth_token
    else:
        auth_token = None

    run_on_cuda: bool = commands.device.startswith("cuda")
    Path(commands.output).mkdir(parents=True, exist_ok=True)
    if run_on_cuda:
        assert torch.cuda.is_available(), "CUDA/GPU is not available on Pytorch. Please check your CUDA installation"
    tokenizer_path = commands.tokenizer if commands.tokenizer else commands.model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=auth_token)
    model_config: PretrainedConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=commands.model, use_auth_token=auth_token
    )
    model_pytorch = AutoModelForSeq2SeqLM.from_pretrained(commands.model, use_auth_token=auth_token)
    input_names = ["input_ids"]
    logging.info(f"axis: {input_names}")

    # generate data for t5 conversion:
    model_pytorch.eval()
    input_ids: torch.Tensor = tokenizer(
        "translate English to French: Transfer learning, where a model is first pre-trained on a data-rich task before"
        " being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing"
        " (NLP).",
        return_tensors=TensorType.PYTORCH,
    ).input_ids
    input_ids = input_ids.type(torch.int32)
    if run_on_cuda:
        model_pytorch.cuda()
        input_ids = input_ids.to("cuda")
    # convert t5 model to onnx format
    """convert_t5_to_onnx(tokenizer=tokenizer, model_pytorch=model_pytorch, path_dir=commands.output, input_ids=input_ids)"""

    timings = {}
    inputs_pytorch: List[Dict[str, Union[np.ndarray, torch.Tensor]]] = [{"input_ids": input_ids}]
    with torch.inference_mode():
        logging.info("running Pytorch (FP32) benchmark")
        engine_name = "Pytorch (FP32)"
        pytorch_output, time_buffer = launch_inference(
            infer=infer_classification_pytorch(model=model_pytorch, run_on_cuda=run_on_cuda, generate_text=True),
            inputs=inputs_pytorch,
            nb_measures=commands.nb_measures,
        )

        timings[engine_name] = time_buffer
        decode_model_outputs(tokenizer, pytorch_output[0][0], engine_name)
        if run_on_cuda and not commands.fast:
            from torch.cuda.amp import autocast

            with autocast():
                engine_name = "Pytorch (FP16)"
                logging.info("running Pytorch (FP16) benchmark")
                pytorch_output, timings[engine_name] = launch_inference(
                    infer=infer_classification_pytorch(model=model_pytorch, run_on_cuda=run_on_cuda, generate_text=True),
                    inputs=inputs_pytorch,
                    nb_measures=commands.nb_measures,
                )
                decode_model_outputs(tokenizer, pytorch_output[0][0], engine_name)
        elif commands.device == "cpu":
            logging.info("preparing Pytorch (INT-8) benchmark")
            model_pytorch = torch.quantization.quantize_dynamic(model_pytorch, {torch.nn.Linear}, dtype=torch.qint8)
            engine_name = "Pytorch (INT-8)"
            pytorch_output, timings[engine_name] = launch_inference(
                infer=infer_classification_pytorch(model=model_pytorch, run_on_cuda=run_on_cuda, generate_text=True),
                inputs=inputs_pytorch,
                nb_measures=commands.nb_measures,
            )
            decode_model_outputs(tokenizer, pytorch_output[0][0], engine_name)
    model_pytorch.cpu()
    logging.info("cleaning up")
    if run_on_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    if "onnx" in commands.backend:
        num_attention_heads, hidden_size = get_model_size(path=commands.model)
        # create optimized onnx model and compare results
        for model_path in [
            os.path.join(commands.output, "t5-encoder") + path for path in ["/model_fp16.onnx", "/model.onnx"]
        ]:
            optim_model_path = model_path[:-5] + "_optim.onnx"
            """optimize_onnx(
                onnx_path=model_path,
                onnx_optim_model_path=optim_model_path,
                fp16=run_on_cuda,
                use_cuda=run_on_cuda,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                architecture=model_config.model_type,
            )"""

        ort_provider = "CUDAExecutionProvider" if run_on_cuda else "CPUExecutionProvider"

        def infer_ort(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
            return ort_model.generate(
                **inputs,
                min_length=10,
                max_length=128,
                num_beams=2,
                no_repeat_ngram_size=2,
            )[0]

        for provider, use_fp16, benchmark_name in [
            (ort_provider, False, "ONNX Runtime (FP32)"),
            (ort_provider, True, "ONNX Runtime (FP16)"),
        ]:
            torch_type = torch.float16 if use_fp16 else torch.float
            encoder_path = os.path.join(commands.output, "t5-encoder") + (
                "/model_fp16.onnx" if use_fp16 else "/model.onnx"
            )
            decoder_path = os.path.join(commands.output, "t5-dec-if-node") + (
                "/model_fp16.onnx" if use_fp16 else "/model.onnx"
            )
            ort_model = (
                ExtT5(
                    config=model_config,
                    device="cuda",
                    encoder_path=encoder_path,
                    decoder_path=decoder_path,
                    torch_type=torch_type,
                )
                .cuda()
                .eval()
            )
            logging.info("running %s benchmark", benchmark_name)
            inputs_pytorch[0]["enable_cache"] = torch.ones(1).type(torch.bool).to("cuda")
            ort_output, time_buffer = launch_inference(infer=infer_ort, inputs=inputs_pytorch, nb_measures=commands.nb_measures)
            decode_model_outputs(tokenizer, ort_output[0], benchmark_name)
            timings[benchmark_name] = time_buffer
            del ort_model
            gc.collect()

            create_triton_configs(
                tokenizer, model_config, pytorch_output, EngineType.ONNX, commands.task, commands.nb_instances,
                input_names, commands.output, commands.device
            )

    if run_on_cuda:
        from torch.cuda import get_device_name

        print(f"Inference done on {get_device_name(0)}")

    print("latencies:")
    for name, time_buffer in timings.items():
        print_timings(name=name, timings=time_buffer)


def entrypoint():
    args = parse_args()
    main(commands=args)


if __name__ == "__main__":
    entrypoint()
