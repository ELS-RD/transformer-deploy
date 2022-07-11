#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import contextlib
import gc
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchdynamo
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from onnxruntime import GraphOptimizationLevel
from tensorrt.tensorrt import Runtime
from torch._C._autograd import ProfilerActivity
from torchdynamo.eval_frame import OptimizeContext
from transformers import PreTrainedModel

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding
from transformer_deploy.backends.trt_utils import load_engine


seq_lengths = [16, 64, 128, 256, 384, 512]
batch_sizes = [1, 8, 16, 32, 64, 128, 256]
shapes_to_test: Dict[int, List[int]] = {b_s: seq_lengths for b_s in batch_sizes}


@dataclass
class BenchmarkOutput:
    latency: float
    output: Dict[str, torch.Tensor]


def get_pytorch_input(size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(2, 1000, size=size, dtype=torch.int32, device="cuda"),
        "attention_mask": torch.ones(size=size, dtype=torch.int32, device="cuda"),
    }


def benchmark(
    fn: Callable[[Dict[str, torch.Tensor]], BenchmarkOutput], shapes: Dict[int, List[int]] = shapes_to_test
) -> np.ndarray:
    gc.collect()  # delete all deletable objects so CUDA memory can be freed by empty_cache
    torch.cuda.empty_cache()
    timings: List[List[float]] = list()
    for is_warmup in [True, True, False]:
        for batch_size, seq_lens in shapes.items():
            batch_timings: List[float] = list()
            for seq_len in seq_lens:
                inputs = get_pytorch_input(size=(batch_size, seq_len))
                latencies = list()
                nb_retry = 1 if is_warmup else 5
                for _ in range(nb_retry):
                    results: BenchmarkOutput = fn(inputs)
                    latencies.append(results.latency)
                batch_timings.append(float(np.median(latencies)))
            if not is_warmup:
                timings.append(batch_timings)
    return np.array(timings)


def get_pytorch_inference(
    model: PreTrainedModel, context_managers: List[contextlib.contextmanager]
) -> Callable[[Dict[str, torch.Tensor]], BenchmarkOutput]:
    def fn(inputs: Dict[str, torch.Tensor]) -> BenchmarkOutput:
        context_managers_stack = contextlib.ExitStack()
        for cm in context_managers:
            context_managers_stack.enter_context(cm)
        with context_managers_stack:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(**inputs)
            torch.cuda.synchronize()
            timing = time.perf_counter() - start
        return BenchmarkOutput(latency=timing, output=output)

    return fn


def get_onnx_inference(
    onnx_path: str, optimization_level: GraphOptimizationLevel
) -> Callable[[Dict[str, torch.Tensor]], BenchmarkOutput]:
    onnx_session = create_model_for_provider(
        onnx_path, provider_to_use="CUDAExecutionProvider", optimization_level=optimization_level
    )
    onnx_binding = onnx_session.io_binding()

    def fn(inputs: Dict[str, torch.Tensor]) -> BenchmarkOutput:
        start = time.perf_counter()
        output = inference_onnx_binding(
            model_onnx=onnx_session, inputs=inputs, device="cuda", binding=onnx_binding, clone_tensor=False
        )
        timing = time.perf_counter() - start
        return BenchmarkOutput(latency=timing, output=output)

    return fn


def get_tensorrt_inference(runtime: Runtime, plan_path: str) -> Callable[[Dict[str, torch.Tensor]], BenchmarkOutput]:
    trt_inference = load_engine(runtime=runtime, engine_file_path=plan_path)

    def fn(inputs: Dict[str, torch.Tensor]) -> BenchmarkOutput:
        start = time.perf_counter()
        output = trt_inference(inputs)
        timing = time.perf_counter() - start
        return BenchmarkOutput(latency=timing, output=output)

    return fn


def check_output(
    fn: Callable[[Dict[str, torch.Tensor]], BenchmarkOutput],
    inputs: Dict[str, torch.Tensor],
    expected_outputs: Dict[str, torch.Tensor],
    atol: int = 1e-1,
) -> None:
    model_output: BenchmarkOutput = fn(inputs)
    for tensor_name in expected_outputs.keys():
        assert model_output.output[tensor_name].shape == expected_outputs[tensor_name].shape
        reference = expected_outputs[tensor_name]
        to_check = model_output.output[tensor_name].type_as(reference)  # to manage the case of float16
        assert torch.allclose(to_check, reference, atol=atol), f"{tensor_name} diff > {atol}"


# for tensorrt there is no way to provide min_shape, opt_shape,max_shape -> dynamic shape requires recompilation
# so better to use the true one
def get_dynamo_optimizer(
    name: str, dynamic_shape: bool = True, dynamo_cache_size: int = 64, reset_cache: bool = True
) -> OptimizeContext:
    # breaks in the graph in small parts, better performance, but no recompilation when size change!
    torchdynamo.config.dynamic_shapes = dynamic_shape
    torchdynamo.config.cache_size_limit = dynamo_cache_size
    if reset_cache:
        torchdynamo.reset()
    # to parameter nvfuser https://github.com/pytorch/pytorch/blob/release/1.12/torch/csrc/jit/codegen/cuda/README.md
    return torchdynamo.optimize(name)  # interesting fusers: nvfuser_ofi, nnc_ofi, fx2trt


def print_pytorch_profile(
    fn: Callable[[Dict[str, torch.Tensor]], BenchmarkOutput], inputs: Dict[str, torch.Tensor], row_limit: int = 20
) -> None:
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA], profile_memory=True, with_flops=True) as prof:
        fn(inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))


def plot_benchmarks(title: str, latencies: np.ndarray, baseline: np.ndarray, batches: List[int] = batch_sizes) -> None:
    sns.set_style("whitegrid")  # darkgrid, whitegrid, dark, white and ticks
    plt.rc("axes", titlesize=15)  # fontsize of the axes title
    plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
    plt.rc("legend", fontsize=15)  # legend fontsize
    plt.rc("font", size=13)  # controls default text sizes

    colors = sns.color_palette("deep")
    batch_latencies = baseline / latencies
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True, sharey=False
    )  # type: Tuple[Figure, Tuple[Axes, Axes]]

    plt.suptitle(f"{title}")
    plt.xticks(seq_lengths)
    ax1.set_title("effect of arithmetic intensity on speedup")
    ax2.set_title("throughput")
    ax1.set_ylabel("speedup over baseline")
    ax2.set_ylabel("# processed sequences per second")
    plt.xlabel("sequence length")

    for i in range(batch_latencies.shape[0]):
        batch = batches[i]
        ax1.plot(seq_lengths, batch_latencies[i], label=batch, color=colors[i])
        ax2.plot(seq_lengths, batch / latencies[i], label=batch, color=colors[i])

    ax1.legend(title="batch size")
    ax2.legend(title="batch size")

    plt.show()
