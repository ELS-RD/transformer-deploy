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
from typing import Callable, Dict, List, Tuple, Type, Union

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from transformer_deploy.backends.ort_utils import (
    cpu_quantization,
    create_model_for_provider,
    inference_onnx_binding,
    optimize_onnx,
)
from transformer_deploy.backends.pytorch_utils import (
    convert_to_onnx,
    get_model_size,
    infer_classification_pytorch,
    infer_feature_extraction_pytorch,
    infer_text_generation,
)
from transformer_deploy.backends.st_utils import STransformerWrapper, load_sentence_transformers
from transformer_deploy.benchmarks.utils import (
    compare_outputs,
    generate_multiple_inputs,
    print_timings,
    setup_logging,
    to_numpy,
    track_infer_time,
)
from transformer_deploy.t5_utils.conversion_utils import (
    ExtT5,
    convert_t5_to_onnx,
    create_triton_configs,
    generate_input_for_t5,
    get_triton_output_shape,
    onnx_to_tensorrt_model,
)
from transformer_deploy.triton.configuration import Configuration, EngineType
from transformer_deploy.triton.configuration_decoder import ConfigurationDec
from transformer_deploy.triton.configuration_encoder import ConfigurationEnc
from transformer_deploy.triton.configuration_question_answering import ConfigurationQuestionAnswering
from transformer_deploy.triton.configuration_token_classifier import ConfigurationTokenClassifier
from transformer_deploy.utils.args import parse_args


def check_accuracy(
    engine_name: str,
    pytorch_output: List[torch.Tensor],
    engine_output: List[Union[np.ndarray, torch.Tensor]],
    tolerance: float,
) -> None:
    """
    Compare engine predictions with a reference.
    Assert that the difference is under a threshold.

    :param engine_name: string used in error message, if any
    :param pytorch_output: reference output used for the comparaison
    :param engine_output: output from the engine
    :param tolerance: if difference in outputs is above threshold, an error will be raised
    """
    pytorch_output = to_numpy(pytorch_output)
    engine_output = to_numpy(engine_output)
    discrepency = compare_outputs(pytorch_output=pytorch_output, engine_output=engine_output)
    assert discrepency <= tolerance, (
        f"{engine_name} discrepency is too high ({discrepency:.2f} >= {tolerance}):\n"
        f"Pythorch:\n{pytorch_output}\n"
        f"VS\n"
        f"Engine:\n{engine_output}\n"
        f"Diff:\n"
        f"{torch.asarray(pytorch_output) - torch.asarray(engine_output)}\n"
        "Tolerance can be increased with --atol parameter."
    )


def launch_inference(
    infer: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    inputs: List[Dict[str, Union[np.ndarray, torch.Tensor]]],
    nb_measures: int,
) -> Tuple[List[Union[np.ndarray, torch.Tensor]], List[float]]:
    """
    Perform inference and measure latency.

    :param infer: a lambda which will perform the inference
    :param inputs: tensor compatible with the lambda (Torch tensor for Pytorch, or numpy otherwise)
    :param nb_measures: number of measures to perform for the latency measure
    :return: a tuple of model output and inference latencies
    """
    assert type(inputs) == list
    assert len(inputs) > 0
    outputs = list()
    for batch_input in inputs:
        output = infer(batch_input)
        outputs.append(output)
    time_buffer: List[int] = list()
    for _ in range(nb_measures):
        with track_infer_time(time_buffer):
            _ = infer(inputs[0])
    return outputs, time_buffer


def main(commands: argparse.Namespace):
    torch.cuda.empty_cache()
    setup_logging(level=logging.INFO if commands.verbose else logging.WARNING)
    logging.info("running with commands: %s", commands)
    # set seeds:
    torch.manual_seed(commands.seed)
    np.random.seed(commands.seed)
    torch.set_num_threads(commands.nb_threads)

    # set device
    if commands.device is None:
        commands.device = "cuda" if torch.cuda.is_available() else "cpu"

    if commands.device == "cpu" and "tensorrt" in commands.backend:
        raise Exception("can't perform inference on CPU and use Nvidia TensorRT as backend")

    if len(commands.seq_len) == len(set(commands.seq_len)) and "tensorrt" in commands.backend:
        logging.warning("having different sequence lengths may make TensorRT slower")

    run_on_cuda: bool = commands.device.startswith("cuda")
    if run_on_cuda:
        assert torch.cuda.is_available(), "CUDA/GPU is not available on Pytorch. Please check your CUDA installation"

    # set authentication
    if isinstance(commands.auth_token, str) and commands.auth_token.lower() in ["true", "t"]:
        auth_token = True
    elif isinstance(commands.auth_token, str):
        auth_token = commands.auth_token
    else:
        auth_token = None

    Path(commands.output).mkdir(parents=True, exist_ok=True)
    tokenizer_path = commands.tokenizer if commands.tokenizer else commands.model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=auth_token)
    model_config: PretrainedConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=commands.model, use_auth_token=auth_token
    )
    input_names: List[str] = tokenizer.model_input_names
    if commands.task == "embedding":
        model_pytorch: Union[PreTrainedModel, STransformerWrapper] = load_sentence_transformers(
            commands.model, use_auth_token=auth_token
        )
    elif commands.task == "classification":
        model_pytorch = AutoModelForSequenceClassification.from_pretrained(commands.model, use_auth_token=auth_token)
    elif commands.task == "token-classification":
        model_pytorch = AutoModelForTokenClassification.from_pretrained(commands.model, use_auth_token=auth_token)
    elif commands.task == "question-answering":
        model_pytorch = AutoModelForQuestionAnswering.from_pretrained(commands.model, use_auth_token=auth_token)
    elif commands.task == "text-generation" and commands.generative_model == "gpt":
        model_pytorch = AutoModelForCausalLM.from_pretrained(commands.model, use_auth_token=auth_token)
        input_names = ["input_ids"]
    elif commands.task == "text-generation" and commands.generative_model == "t5":
        model_pytorch = AutoModelForSeq2SeqLM.from_pretrained(commands.model, use_auth_token=auth_token)
        input_names = ["input_ids"]
    else:
        raise Exception(f"unknown task: {commands.task}")

    if hasattr(model_config, "type_vocab_size") and model_config.type_vocab_size == 0:
        try:
            input_names.remove("token_type_ids")
            logging.warning("Model doesn't have `token_type_ids`, removing them from `input_names`")
        except ValueError:
            pass

    logging.info(f"axis: {input_names}")

    model_pytorch.eval()
    if run_on_cuda:
        model_pytorch.cuda()

    tensor_shapes = list(zip(commands.batch_size, commands.seq_len))
    # create onnx model and compare results
    if commands.task == "text-generation" and commands.generative_model == "t5":
        input_ids = generate_input_for_t5(tokenizer, run_on_cuda)
        inputs_pytorch: List[Dict[str, Union[np.ndarray, torch.Tensor]]] = [{"input_ids": input_ids}]
        convert_t5_to_onnx(
            tokenizer=tokenizer,
            model_pytorch=model_pytorch,
            path_dir=commands.output,
            input_ids=input_ids,
        )
    else:
        onnx_model_path = os.path.join(commands.output, "model-original.onnx")
        # take optimal size
        inputs_pytorch = generate_multiple_inputs(
            batch_size=tensor_shapes[1][0],
            seq_len=tensor_shapes[1][1],
            input_names=input_names,
            device=commands.device,
            nb_inputs_to_gen=commands.warmup,
        )
        convert_to_onnx(
            model_pytorch=model_pytorch,
            output_path=onnx_model_path,
            inputs_pytorch=inputs_pytorch[0],
            quantization=commands.quantization,
            var_output_seq=commands.task in ["text-generation", "token-classification", "question-answering"],
            output_names=["output"] if commands.task != "question-answering" else ["start_logits", "end_logits"],
        )

    timings = {}

    def get_pytorch_infer(model: PreTrainedModel, cuda: bool, task: str):
        if task == "text-generation" and commands.generative_model == "t5":
            return infer_text_generation(
                model=model,
                run_on_cuda=cuda,
                min_length=commands.seq_len[0],
                max_length=commands.seq_len[1],
                num_beams=2,
            )
        if task in ["classification", "text-generation", "token-classification", "question-answering"]:
            return infer_classification_pytorch(model=model, run_on_cuda=cuda)
        if task == "embedding":
            return infer_feature_extraction_pytorch(model=model, run_on_cuda=cuda)
        raise Exception(f"unknown task: {task}")

    with torch.inference_mode():
        logging.info("running Pytorch (FP32) benchmark")
        pytorch_output, time_buffer = launch_inference(
            infer=get_pytorch_infer(model=model_pytorch, cuda=run_on_cuda, task=commands.task),
            inputs=inputs_pytorch,
            nb_measures=commands.nb_measures,
        )
        timings["Pytorch (FP32)"] = time_buffer
        if run_on_cuda and not commands.fast:
            from torch.cuda.amp import autocast

            with autocast():
                engine_name = "Pytorch (FP16)"
                logging.info("running Pytorch (FP16) benchmark")
                pytorch_fp16_output, time_buffer = launch_inference(
                    infer=get_pytorch_infer(model=model_pytorch, cuda=run_on_cuda, task=commands.task),
                    inputs=inputs_pytorch,
                    nb_measures=commands.nb_measures,
                )
                check_accuracy(
                    engine_name=engine_name,
                    pytorch_output=pytorch_output,
                    engine_output=pytorch_fp16_output,
                    tolerance=commands.atol,
                )
                timings[engine_name] = time_buffer
        elif commands.device == "cpu":
            logging.info("preparing Pytorch (INT-8) benchmark")
            model_pytorch = torch.quantization.quantize_dynamic(model_pytorch, {torch.nn.Linear}, dtype=torch.qint8)
            engine_name = "Pytorch (INT-8)"
            logging.info("running Pytorch (FP32) benchmark")
            pytorch_int8_output, time_buffer = launch_inference(
                infer=get_pytorch_infer(model=model_pytorch, cuda=run_on_cuda, task=commands.task),
                inputs=inputs_pytorch,
                nb_measures=commands.nb_measures,
            )
            check_accuracy(
                engine_name=engine_name,
                pytorch_output=pytorch_output,
                engine_output=pytorch_int8_output,
                tolerance=commands.atol,
            )
            timings[engine_name] = time_buffer

    # create triton conf for models different from T5
    if commands.generative_model != "t5":
        if commands.task == "text-generation" and commands.generative_model == "gpt":
            conf_class: Type[Configuration] = ConfigurationDec
        elif commands.task == "token-classification":
            conf_class: Type[Configuration] = ConfigurationTokenClassifier
        elif commands.task == "question-answering":
            conf_class: Type[Configuration] = ConfigurationQuestionAnswering
        else:
            conf_class = ConfigurationEnc

        triton_conf = conf_class(
            model_name_base=commands.name,
            dim_output=get_triton_output_shape(
                output=pytorch_output[0] if type(pytorch_output[0]) == torch.Tensor else pytorch_output[0][0],
                task=commands.task,
            ),
            nb_instance=commands.nb_instances,
            tensor_input_names=input_names,
            working_directory=commands.output,
            device=commands.device,
        )
    model_pytorch.cpu()

    logging.info("cleaning up")
    if run_on_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    if "tensorrt" in commands.backend:
        logging.info("preparing TensorRT (FP16) benchmark")
        try:
            import tensorrt as trt
            from tensorrt.tensorrt import Logger, Runtime

            from transformer_deploy.t5_utils.conversion_utils import prepare_input_shapes_tensorrt_decoder

        except ImportError:
            raise ImportError(
                "It seems that TensorRT is not yet installed. "
                "It is required when you declare TensorRT backend."
                "Please find installation instruction on "
                "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
            )
        trt_logger: Logger = trt.Logger(trt.Logger.VERBOSE if commands.verbose else trt.Logger.WARNING)
        runtime: Runtime = trt.Runtime(trt_logger)
        if commands.generative_model == "t5":
            encoder_onnx_path = os.path.join(commands.output, "t5-encoder") + "/model.onnx"
            tensorrt_encoder_path = os.path.join(commands.output, "t5-encoder") + "/model.plan"
            tensorrt_encoder = onnx_to_tensorrt_model(
                runtime=runtime,
                onnx_model_path=encoder_onnx_path,
                trt_logger=trt_logger,
                tensor_shapes=tensor_shapes,
                workspace_size=commands.workspace_size,
                quantization=commands.quantization,
                tensorrt_model_path=tensorrt_encoder_path,
            )

            decoder_onnx_path = os.path.join(commands.output, "t5-dec-if-node") + "/model.onnx"
            tensorrt_decoder_path = os.path.join(commands.output, "t5-dec-if-node") + "/model.plan"
            input_shapes = prepare_input_shapes_tensorrt_decoder(input_ids, model_pytorch.config.num_layers)
            tensorrt_decoder = onnx_to_tensorrt_model(
                runtime=runtime,
                onnx_model_path=decoder_onnx_path,
                trt_logger=trt_logger,
                tensor_shapes=input_shapes,
                workspace_size=commands.workspace_size,
                quantization=commands.quantization,
                tensorrt_model_path=tensorrt_decoder_path,
            )

            def tensorrt_model(x):
                encoder_outputs = tensorrt_encoder(x)
                inputs = {
                    "input_ids": input_ids,
                    "encoder_hidden_states": encoder_outputs.last_hidden_state,
                    "final_seq_len": torch.tensor([1], dtype=torch.int32, device="cuda"),
                    "enable_cache": torch.tensor([True], dtype=torch.bool, device="cuda"),
                }
                decoder_outputs = tensorrt_decoder(inputs)
                return decoder_outputs[0]

        else:
            tensorrt_path = os.path.join(commands.output, "model.plan")
            tensorrt_model = onnx_to_tensorrt_model(
                runtime=runtime,
                onnx_model_path=onnx_model_path,
                trt_logger=trt_logger,
                tensor_shapes=tensor_shapes,
                workspace_size=commands.workspace_size,
                quantization=commands.quantization,
                tensorrt_model_path=tensorrt_path,
            )

        if commands.task == "question-answering":
            tensorrt_inf: Callable[[Dict[str, torch.Tensor]], List[torch.Tensor]] = lambda x: list(
                tensorrt_model(x).values()
            )
        else:
            tensorrt_inf: Callable[[Dict[str, torch.Tensor]], torch.Tensor] = lambda x: list(
                tensorrt_model(x).values()
            )[0]

        logging.info("running TensorRT (FP16) benchmark")
        engine_name = "TensorRT (FP16)"
        tensorrt_output, time_buffer = launch_inference(
            infer=tensorrt_inf, inputs=inputs_pytorch, nb_measures=commands.nb_measures
        )
        check_accuracy(
            engine_name=engine_name,
            pytorch_output=pytorch_output,
            engine_output=tensorrt_output,
            tolerance=commands.atol,
        )
        timings[engine_name] = time_buffer
        del tensorrt_model, runtime  # delete all tensorrt objects
        gc.collect()
        if commands.task == "text-generation" and commands.generative_model == "t5":
            encoder_outputs = tensorrt_encoder(input_ids)
            create_triton_configs(
                tokenizer,
                model_config,
                encoder_outputs,
                tensorrt_output,
                EngineType.TensorRT,
                commands.task,
                commands.nb_instances,
                input_names,
                commands.output,
                commands.device,
            )
        else:
            triton_conf.create_configs(
                tokenizer=tokenizer, model_path=tensorrt_path, config=model_config, engine_type=EngineType.TensorRT
            )

    if "onnx" in commands.backend:
        # create optimized onnx model and compare results
        num_attention_heads, hidden_size = get_model_size(path=commands.model, auth_token=auth_token)
        model_paths = (
            [os.path.join(commands.output, "t5-encoder") + path for path in ["/model_fp16.onnx", "/model.onnx"]]
            if commands.generative_model == "t5"
            else [onnx_model_path]
        )
        optim_model_paths = (
            [model_path[:-5] + "_optim.onnx" for model_path in model_paths]
            if commands.generative_model == "t5"
            else [os.path.join(commands.output, "model.onnx")]
        )
        [
            optimize_onnx(
                onnx_path=model_paths[idx],
                onnx_optim_model_path=optim_model_path,
                fp16=run_on_cuda,
                use_cuda=run_on_cuda,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                architecture=model_config.model_type,
            )
            for idx, optim_model_path in enumerate(optim_model_paths)
        ]

        if commands.device == "cpu" and commands.quantization:
            cpu_quantization(input_model_path=optim_model_paths[0], output_model_path=optim_model_paths[0])

        ort_provider = "CUDAExecutionProvider" if run_on_cuda else "CPUExecutionProvider"
        for provider, is_fp16, benchmark_name in [
            (ort_provider, False, "ONNX Runtime (FP32)"),
            (ort_provider, True, "ONNX Runtime (FP16)"),
        ]:
            logging.info("preparing %s benchmark", benchmark_name)
            torch_type = torch.float16 if is_fp16 else torch.float32
            if commands.generative_model == "t5":
                encoder_path = os.path.join(commands.output, "t5-encoder") + (
                    "/model_fp16.onnx" if is_fp16 else "/model.onnx"
                )
                decoder_path = os.path.join(commands.output, "t5-dec-if-node") + (
                    "/model_fp16.onnx" if is_fp16 else "/model.onnx"
                )
                ort_model = (
                    ExtT5(
                        config=model_pytorch.config,
                        device="cuda",
                        encoder_path=encoder_path,
                        decoder_path=decoder_path,
                        torch_type=torch_type,
                    )
                    .cuda()
                    .eval()
                )
                # warmup generative model:
                [
                    ort_model.generate(
                        inputs=input_ids, min_length=commands.seq_len[0], max_length=commands.seq_len[1], num_beams=2
                    )
                    for _ in range(5)
                ]
            else:
                model_path = onnx_model_path if is_fp16 else optim_model_paths[0]
                ort_model = create_model_for_provider(
                    path=model_path,
                    provider_to_use=provider,
                    nb_threads=commands.nb_threads,
                )

            def infer_ort(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
                results = (
                    ort_model.generate(
                        inputs=inputs,
                        min_length=commands.seq_len[0],
                        max_length=commands.seq_len[1],
                        num_beams=2,
                    )[0]
                    if commands.generative_model == "t5"
                    else inference_onnx_binding(model_onnx=ort_model, inputs=inputs, device=commands.device)
                )
                if commands.generative_model == "t5":
                    return results
                elif "output" in results:
                    return results["output"]
                else:
                    return results["start_logits"], results["end_logits"]

            logging.info("running %s benchmark", benchmark_name)
            inputs = [input_ids] if commands.generative_model == "t5" else inputs_pytorch
            ort_output, time_buffer = launch_inference(infer=infer_ort, inputs=inputs, nb_measures=commands.nb_measures)
            check_accuracy(
                engine_name=benchmark_name,
                pytorch_output=pytorch_output[0] if commands.generative_model == "t5" else pytorch_output,
                engine_output=ort_output,
                tolerance=100000,
            )
            timings[benchmark_name] = time_buffer
            gc.collect()

        if commands.generative_model == "t5":
            encoder_output = ort_model.get_encoder()(input_ids)
            decoder_output = ort_model.forward(
                input_ids, encoder_output.last_hidden_state, torch.tensor([0], dtype=torch.int32, device="cuda"), None
            )
            create_triton_configs(
                tokenizer,
                model_config,
                encoder_output,
                decoder_output,
                EngineType.ONNX,
                commands.task,
                commands.nb_instances,
                input_names,
                commands.output,
                commands.device,
            )
            del ort_model, encoder_output, decoder_output
        else:
            triton_conf.create_configs(
                tokenizer=tokenizer,
                model_path=optim_model_paths[0],
                config=model_config,
                engine_type=EngineType.ONNX,
            )

    if run_on_cuda:
        from torch.cuda import get_device_name

        print(f"Inference done on {get_device_name(0)}")

    print("latencies:")
    for name, time_buffer in timings.items():
        print_timings(name=name, timings=time_buffer)
    print(f"Each inference engine output is within {commands.atol} tolerance compared to Pytorch output")


def entrypoint():
    args = parse_args()
    main(commands=args)


if __name__ == "__main__":
    entrypoint()
