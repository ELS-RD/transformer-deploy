import argparse
import logging
import os
from pathlib import Path
from typing import List, Callable, Union

import numpy as np
import torch
from pycuda._driver import Stream
from torch.cuda import get_device_name
from torch.cuda.amp import autocast
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel, AutoTokenizer,
)

from backends.ort_utils import create_model_for_provider, convert_to_onnx, optimize_onnx
from backends.trt_utils import build_engine, save_engine, get_binding_idxs, infer_tensorrt
from benchmarks.utils import print_timings, setup_logging, track_infer_time, prepare_input
from tensorrt.tensorrt import Logger, Runtime, IExecutionContext
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TODO choose the best model
# TODO create folder structure (to save somewhere files first)
# block if folder already exist, add parameter to overwrite a folder
# TODO adapt README to show command line to type to launch and benchmark the server
# TODO script shell to run all commands including the benchmark
# TODO format code
# TODO add verbose mode

torch.manual_seed(123)
setup_logging()


def main():
    parser = argparse.ArgumentParser(description="optimize and deploy transformers", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="path to model or URL to Hugging Face Hub")
    parser.add_argument("-b", "--batch-size", default=[1, 1, 1], help="batch sizes to optimize for (min, opt, max)", type=int, nargs=3)
    parser.add_argument("-s", "--seq-len", default=[16, 16, 16], help="sequence lengths to optimize for (min, opt, max)", type=int, nargs=3)
    parser.add_argument("-w", "--workspace-size", default=10000, help="workspace size in MiB (TensorRT)", type=int)
    parser.add_argument("-o", "--output", default="triton_models", help="name to be used for ")
    parser.add_argument("-n", "--name", default="transformer", help="name to be used in triton server")
    parser.add_argument("-p", "--pytorch", action="store_true",  help="include Pytorch measures")
    parser.add_argument("--warmup", default=100, help="# of inferences to warm each model", type=int)
    parser.add_argument("--nb-measures", default=1000, help="# of inferences for benchmarks", type=int)
    args, _ = parser.parse_known_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    onnx_model_path = os.path.join(args.output, "model-original.onnx")
    onnx_optim_fp16_path = os.path.join(args.output, "model.onnx")
    tensorrt_path = os.path.join(args.output, "model.plan")

    assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_names: List[str] = tokenizer.model_input_names
    model_pytorch: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(args.model)
    model_pytorch.cuda()
    model_pytorch.eval()

    tensor_shapes = list(zip(args.batch_size, args.seq_len))
    inputs_pytorch, inputs_onnx = prepare_input(batch_size=tensor_shapes[-1][0], seq_len=tensor_shapes[-1][1], include_token_ids="token_type_ids" in input_names)

    with torch.inference_mode():
        output = model_pytorch(**inputs_pytorch)
        output = output.logits  # extract the value of interest
        output_pytorch: np.ndarray = output.detach().cpu().numpy()

    logging.info(f"[Pytorch] input shape {inputs_pytorch['input_ids'].shape}")
    logging.info(f"[Pytorch] output shape: {output_pytorch.shape}")
    # create onnx model and compare results
    convert_to_onnx(model_pytorch=model_pytorch, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch)
    onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use="CUDAExecutionProvider")
    output_onnx = onnx_model.run(None, inputs_onnx)

    del onnx_model

    assert np.allclose(a=output_onnx, b=output_pytorch, atol=1e-1)
    if not args.pytorch:
        del model_pytorch

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
    assert np.allclose(a=output_onnx_optimised, b=output_pytorch, atol=1e-1)

    trt_logger: Logger = trt.Logger(trt.Logger.INFO)
    runtime: Runtime = trt.Runtime(trt_logger)
    engine = build_engine(runtime=runtime, onnx_file_path=onnx_model_path, logger=trt_logger, min_shape=tensor_shapes[0], optimal_shape=tensor_shapes[1], max_shape=tensor_shapes[2], workspace_size=args.workspace_size * 1024 * 1024)
    save_engine(engine=engine, engine_file_path=tensorrt_path)
    stream: Stream = pycuda.driver.Stream()
    context: IExecutionContext = engine.create_execution_context()
    profile_index = 0
    context.set_optimization_profile_async(profile_index=profile_index, stream_handle=stream.handle)
    # retrieve input/output IDs
    input_binding_idxs, output_binding_idxs = get_binding_idxs(engine, profile_index)  # type: List[int], List[int]
    tensorrt_output = infer_tensorrt(context=context, host_inputs=inputs_onnx, input_binding_idxs=input_binding_idxs, output_binding_idxs=output_binding_idxs, stream=stream)
    assert np.allclose(a=tensorrt_output, b=output_pytorch, atol=1e-1), f"tensorrt accuracy is too low:\n" \
                                                                        f"Pythorch:\n{output_pytorch}\n" \
                                                                        f"VS\n" \
                                                                        f"TensorRT:\n{tensorrt_output}"

    timings = {}
    for _ in range(args.warmup):
        _ = infer_tensorrt(context=context, host_inputs=inputs_onnx, input_binding_idxs=input_binding_idxs, output_binding_idxs=output_binding_idxs, stream=stream)
    time_buffer = list()
    for _ in range(args.nb_measures):
        with track_infer_time(time_buffer):
            _ = infer_tensorrt(context=context, host_inputs=inputs_onnx, input_binding_idxs=input_binding_idxs, output_binding_idxs=output_binding_idxs, stream=stream)
        timings["tensorrt_fp16"] = time_buffer
    del engine, context, runtime  # delete all tensorrt objects

    providers = [
        ("CUDAExecutionProvider", onnx_model_path),
        ("CUDAExecutionProvider", onnx_optim_fp16_path),
    ]

    for provider, model_path in providers:
        model = create_model_for_provider(path=model_path, provider_to_use=provider)
        time_buffer = []
        for _ in range(args.warmup):
            _ = model.run(None, inputs_onnx)
        for _ in range(args.nb_measures):
            with track_infer_time(time_buffer):
                _ = model.run(None, inputs_onnx)
        timings[f"[{provider}] {model_path}"] = time_buffer
    del model

    if args.pytorch:
        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = model_pytorch(**inputs_pytorch)
                torch.cuda.synchronize()
            time_buffer = []
            for _ in range(args.nb_measures):
                with track_infer_time(time_buffer):
                    _ = model_pytorch(**inputs_pytorch)
                    torch.cuda.synchronize()
            timings["Pytorch_fp32"] = time_buffer
            with autocast():
                for _ in range(args.warmup):
                    _ = model_pytorch(**inputs_pytorch)
                    torch.cuda.synchronize()
                time_buffer = []
                for _ in range(args.nb_measures):
                    with track_infer_time(time_buffer):
                        _ = model_pytorch(**inputs_pytorch)
                        torch.cuda.synchronize()
                timings["Pytorch_fp16"] = time_buffer

    logging.info(f"inference done on {get_device_name(0)}")

    for name, time_buffer in timings.items():
        print_timings(name=name, timings=time_buffer)


if __name__ == "__main__":
    main()
