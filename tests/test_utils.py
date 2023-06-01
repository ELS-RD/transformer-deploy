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
from typing import List, Tuple, Union

import numpy as np
import torch
import transformers.models.t5.modeling_t5

from transformer_deploy.backends.pytorch_utils import get_model_size
from transformer_deploy.benchmarks.utils import compare_outputs, generate_input, generate_multiple_inputs, to_numpy
from transformer_deploy.utils.accuracy import check_accuracy
from transformer_deploy.utils.fastseq import code_patcher


def generate_fake_outputs(
    shape: Tuple[int, int], nb: int, factor: float, tensor_type: str
) -> List[Union[np.ndarray, torch.Tensor]]:
    results = list()
    for _ in range(nb):
        if tensor_type == "np":
            tensor = np.arange(start=0, stop=shape[0] * shape[1]).reshape(shape) * factor
        elif tensor_type == "torch":
            tensor = torch.arange(start=0, end=shape[0] * shape[1], device="cpu").reshape(shape) * factor
        else:
            raise Exception(f"unknown: {tensor_type}")
        results.append(tensor)
    return results


def test_gap():
    shape = (1, 4)
    pairs = [("np", "np"), ("np", "torch"), ("torch", "np"), ("torch", "torch")]
    for t1_type, t2_type in pairs:
        t1 = generate_fake_outputs(shape=shape, nb=1, factor=0.1, tensor_type=t1_type)
        t2 = generate_fake_outputs(shape=shape, nb=1, factor=0.2, tensor_type=t2_type)
        assert np.isclose(a=compare_outputs(pytorch_output=to_numpy(t1), engine_output=to_numpy(t2)), b=0.15, atol=1e-3)
        check_accuracy(engine_name=f"test [{t1_type}/{t2_type}]", pytorch_output=t1, engine_output=t2, tolerance=0.16)


def test_generate_input():
    inputs_pytorch = generate_input(seq_len=16, batch_size=4, input_names=["input_ids", "attention_mask"], device="cpu")
    assert set(inputs_pytorch.keys()) == {"input_ids", "attention_mask"}
    assert inputs_pytorch["input_ids"].shape == torch.Size([4, 16])
    inputs_pytorch = generate_input(
        seq_len=1, batch_size=1, input_names=["input_ids", "attention_mask", "token_type_ids"], device="cpu"
    )
    assert set(inputs_pytorch.keys()) == {"input_ids", "attention_mask", "token_type_ids"}


def test_multiple_generate_input():
    multiple_inputs_pytorch = generate_multiple_inputs(
        seq_len=16, batch_size=4, input_names=["input_ids", "attention_mask"], nb_inputs_to_gen=4, device="cpu"
    )
    assert len(multiple_inputs_pytorch) == 4
    assert set(multiple_inputs_pytorch[0].keys()) == {"input_ids", "attention_mask"}


def test_extract_model_info():
    models = [
        "philschmid/MiniLM-L6-H384-uncased-sst2",
        "camembert-base",
        "sentence-transformers/msmarco-distilbert-cos-v5",
    ]
    for m in models:
        att, hidden_size = get_model_size(path=m)
        assert att > 0 and hidden_size > 0


def test_update_module():
    src_code = (
        "vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}\n"
        "        vocab.update(self.added_tokens_encoder)\n"
    )
    code_patcher(
        module_name="transformers.models.t5.tokenization_t5",
        function=transformers.models.t5.tokenization_t5.T5Tokenizer.get_vocab,
        new_function_name="new_vocab",
        modifications={src_code: 'vocab = {"1": "success"}\n'},
    )
    transformers.models.t5.tokenization_t5.T5Tokenizer.get_vocab = transformers.models.t5.tokenization_t5.new_vocab
    assert transformers.models.t5.tokenization_t5.T5Tokenizer.get_vocab(1) == {"1": "success"}
