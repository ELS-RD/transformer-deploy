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
Command line args parser
"""

import argparse
from typing import List


def parse_args(commands: List[str] = None) -> argparse.Namespace:
    """
    Parse command line arguments
    :param commands: to provide command line programatically
    :return: parsed command line
    """
    parser = argparse.ArgumentParser(
        description="optimize and deploy transformers", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model", required=True, help="path to model or URL to Hugging Face hub")
    parser.add_argument("-t", "--tokenizer", help="path to tokenizer or URL to Hugging Face hub")
    parser.add_argument(
        "--task",
        default="classification",
        choices=["classification", "embedding", "text-generation", "token-classification", "question-answering"],
        help="task to manage. embeddings is for sentence-transformers models",
    )
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
        help="batch sizes to optimize for (min, optimal, max). Used by TensorRT and benchmarks.",
        type=int,
        nargs=3,
    )
    parser.add_argument(
        "-s",
        "--seq-len",
        default=[16, 16, 16],
        help="sequence lengths to optimize for (min, optimal, max). Used by TensorRT and benchmarks.",
        type=int,
        nargs=3,
    )
    parser.add_argument("-q", "--quantization", action="store_true", help="INT-8 GPU quantization support")
    parser.add_argument("-w", "--workspace-size", default=10000, help="workspace size in MiB (TensorRT)", type=int)
    parser.add_argument("-o", "--output", default="triton_models", help="name to be used for ")
    parser.add_argument("-n", "--name", default="transformer", help="model name to be used in triton server")
    parser.add_argument("-v", "--verbose", action="store_true", help="display detailed information")
    parser.add_argument("--fast", action="store_true", help="skip the Pytorch (FP16) benchmark")
    parser.add_argument(
        "--backend",
        default=["onnx"],
        help="backend to use. multiple args accepted.",
        nargs="*",
        choices=["onnx", "tensorrt"],
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        help="device to use. If not set, will be cuda if available.",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--nb-threads", default=1, help="# of CPU threads to use for inference", type=int)
    parser.add_argument(
        "--nb-instances", default=1, help="# of model instances, may improve throughput (Triton)", type=int
    )
    parser.add_argument("--warmup", default=10, help="# of inferences to warm each model", type=int)
    parser.add_argument("--nb-measures", default=1000, help="# of inferences for benchmarks", type=int)
    parser.add_argument("--seed", default=123, help="seed for random inputs, etc.", type=int)
    parser.add_argument("--atol", default=3e-1, help="tolerance when comparing outputs to Pytorch ones", type=float)
    args, _ = parser.parse_known_args(args=commands)
    return args
