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
import tempfile

import pytest

from transformer_deploy.convert import main
from transformer_deploy.utils.args import parse_args


@pytest.mark.gpu
def test_albert_gpu():
    commands = [
        "--model",
        "nreimers/albert-small-v2",
        "--backend",
        "tensorrt",
        "onnx",
        "--batch",
        "1",
        "16",
        "16",
        "--seq-len",
        "8",
        "8",
        "8",
        "--output",
        tempfile.mkdtemp(),
    ]
    args = parse_args(commands=commands)
    main(commands=args)


def test_minilm_cpu():
    commands = [
        "--model",
        "philschmid/MiniLM-L6-H384-uncased-sst2",
        "--backend",
        "onnx",
        "--batch",
        "1",
        "1",
        "1",
        "--seq-len",
        "8",
        "8",
        "8",
        "--device",
        "cpu",
        "--warmup",
        "5",
        "--nb-measures",
        "10",
        "--nb-threads",
        "2",
        "--output",
        tempfile.mkdtemp(),
    ]
    args = parse_args(commands=commands)
    main(commands=args)


@pytest.mark.gpu
def test_minilm_quantization():
    commands = [
        "--model",
        "philschmid/MiniLM-L6-H384-uncased-sst2",
        "--backend",
        "onnx",
        "--batch",
        "1",
        "1",
        "1",
        "--seq-len",
        "8",
        "8",
        "8",
        "--warmup",
        "5",
        "--nb-measures",
        "10",
        "--nb-threads",
        "2",
        "--quantization",
        "--output",
        tempfile.mkdtemp(),
    ]
    args = parse_args(commands=commands)
    main(commands=args)


@pytest.mark.gpu
def test_camembert_gpu():
    commands = [
        "--model",
        "camembert-base",
        "--backend",
        "tensorrt",
        "onnx",
        "--batch",
        "1",
        "16",
        "16",
        "--seq-len",
        "8",
        "8",
        "8",
        "--output",
        tempfile.mkdtemp(),
    ]
    args = parse_args(commands=commands)
    main(commands=args)


@pytest.mark.gpu
def test_electra_gpu():
    commands = [
        "--model",
        "google/electra-small-discriminator",
        "--backend",
        "tensorrt",
        "onnx",
        "--batch",
        "1",
        "16",
        "16",
        "--seq-len",
        "8",
        "8",
        "8",
        "--output",
        tempfile.mkdtemp(),
    ]
    args = parse_args(commands=commands)
    main(commands=args)


# FAILED tests/test_models.py::test_sentence_transformers_cpu - RuntimeError: Error in execution: Non-zero status
# code returned while running EmbedLayerNormalization node. Name:'EmbedLayerNormalization_0' Status
# Message: input_ids and position_ids shall have same shape
def test_sentence_transformers_cpu():
    commands = [
        "--model",
        "sentence-transformers/all-MiniLM-L6-v2",
        "--backend",
        "onnx",
        "--task",
        "embedding",
        "--batch",
        "1",
        "16",
        "16",
        "--seq-len",
        "8",
        "8",
        "8",
        "--device",
        "cpu",
        "--output",
        tempfile.mkdtemp(),
    ]
    args = parse_args(commands=commands)
    main(commands=args)


@pytest.mark.gpu
def test_gpt2_gpu():
    commands = [
        "--model",
        "distilgpt2",
        "--task",
        "text-generation",
        "--backend",
        "onnx",
        "--batch",
        "1",
        "16",
        "16",
        "--seq-len",
        "8",
        "8",
        "8",
        "--output",
        tempfile.mkdtemp(),
    ]
    args = parse_args(commands=commands)
    main(commands=args)


def test_bert_ner_cpu():
    commands = [
        "--model",
        "kamalkraj/bert-base-cased-ner-conll2003",
        "--task",
        "token-classification",
        "--backend",
        "onnx",
        "--batch",
        "1",
        "16",
        "16",
        "--seq-len",
        "8",
        "8",
        "8",
        "--output",
        tempfile.mkdtemp(),
    ]
    args = parse_args(commands=commands)
    main(commands=args)
