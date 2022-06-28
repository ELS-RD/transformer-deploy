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
import time
from typing import List, Tuple, Union

import numpy as np
import torch
import transformers.models.t5.modeling_t5
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, TensorType

from transformer_deploy.backends.pytorch_utils import get_model_size
from transformer_deploy.benchmarks.utils import compare_outputs, generate_input, generate_multiple_inputs, to_numpy
from transformer_deploy.convert import check_accuracy
from transformer_deploy.utils.code_utils import update_module


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


def generate_texts(
    model,
    tokenizer,
    batch_count,
    slines,
    max_token_length: int = 1024,
    num_beams: int = 4,
    min_gen_length: int = 55,
    max_gen_length: int = 199,
    no_repeat_ngram_size: int = 3,
    early_stopping: bool = True,
    use_cache: bool = True,
    do_log=True,
):
    """Generate the summaries.

    Args:
        slines (List(str)): a list of input sentences.
        max_token_length (int): max tokenized sentence length.
        num_beams (int): beam number.
        min_gen_length (int): min generation length.
        max_gen_length (int): maxium length for the generation output.
        no_repeat_ngram_size (int): size of no repeat gram.
        early_stopping (bool): indicate if the beam search will be early
            stopped.
        use_cache (bool): If `use_cache` is True, past key values are used
            to speed up decoding if applicable to model.

    Returns:
        List(str): a list of generated summaries.
    """
    start = time.time()
    with torch.no_grad():
        inputs = tokenizer(slines, max_length=max_token_length, padding=True, truncation=True, return_tensors="pt")

        # Generate Summary
        summary_ids = model.generate(
            inputs["input_ids"].cuda(),
            num_beams=num_beams,
            min_length=min_gen_length,
            max_length=max_gen_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            use_cache=use_cache,
        )
        outputs = [tokenizer.decode(g) for g in summary_ids]
    end = time.time()
    if do_log:
        print(f"batch-{batch_count}: Process {len(slines)} samples in {end - start:.2f} seconds")
    return outputs


def transformers_modifications_test(
    modify_transformers: bool,
    source_path: str,
    expected_output_path: str,
    nbr_examples: int = 100,
    batch_size: int = 8,
    max_token_length: int = 1024,
    num_beams: int = 4,
    min_gen_length: int = 55,
    max_gen_length: int = 199,
    no_repeat_ngram_size: int = 3,
    early_stopping: bool = True,
    use_cache: bool = True,
):
    test_name = "with transformers modifications" if modify_transformers else "without transformers modifications"
    print(f"Start test {test_name}")
    if modify_transformers:
        update_transformers()
    expected_outputs = []
    with open(expected_output_path, "rt", encoding="utf-8") as expected_output_file:
        for idx, line in enumerate(expected_output_file):
            expected_outputs.append(line.strip())
            if idx + 1 == nbr_examples:
                break
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    model: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.cuda()
    model.eval()
    processed_sample_count = 0
    outputs = []
    slines = []
    start = time.time()
    batch_count = 0
    with open(source_path, "rt", encoding="utf-8") as source:
        for s_idx, sline in enumerate(source):
            slines.append(sline)
            if len(slines) % batch_size:
                if s_idx == nbr_examples:
                    break
                continue
            if batch_count == 0:
                # warmup model:
                generate_texts(
                    model,
                    tokenizer,
                    batch_count,
                    slines,
                    max_token_length,
                    num_beams,
                    min_gen_length,
                    max_gen_length,
                    no_repeat_ngram_size,
                    early_stopping,
                    use_cache,
                    do_log=False,
                )
            outputs.extend(
                generate_texts(
                    model,
                    tokenizer,
                    batch_count,
                    slines,
                    max_token_length,
                    num_beams,
                    min_gen_length,
                    max_gen_length,
                    no_repeat_ngram_size,
                    early_stopping,
                    use_cache,
                )
            )
            processed_sample_count += len(slines)
            slines = []
            batch_count += 1
            if s_idx + 1 == nbr_examples:
                break

        if slines:
            outputs.extend(
                generate_texts(
                    model,
                    tokenizer,
                    batch_count,
                    slines,
                    max_token_length,
                    num_beams,
                    min_gen_length,
                    max_gen_length,
                    no_repeat_ngram_size,
                    early_stopping,
                    use_cache,
                    do_log=False,
                )
            )
            processed_sample_count += len(slines)

        end = time.time()
        samples_per_second = processed_sample_count / (end - start)
        print(
            f"Finish the processing of {processed_sample_count} samples with the speed {samples_per_second:.2f} "
            f"samples/second. Full time: {end - start} "
        )

    for i, output in enumerate(outputs):
        if output != expected_outputs[i]:
            pass


def test_transformers_utils():
    source_path = "./data/cnndm_128.txt"
    expected_output_path = "./data/expected_t5_output.hypo"
    # test with transformers modifications
    modify_transformers = False
    transformers_modifications_test(
        modify_transformers,
        source_path,
        expected_output_path,
        nbr_examples=16,
    )
    # test without transformers modifications
    modify_transformers = True
    transformers_modifications_test(
        modify_transformers,
        source_path,
        expected_output_path,
        nbr_examples=16,
    )


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
    # replace the whole function body
    update_module(
        module_name="transformers.models.t5.modeling_t5",
        function=transformers.models.t5.modeling_t5.T5Attention.forward,
        new_function_name="updatedForward",
        modifications={"*": "return True"},
    )

    transformers.models.t5.modeling_t5.T5Attention.forward = transformers.models.t5.modeling_t5.updatedForward
    assert transformers.models.t5.modeling_t5.T5Attention.forward(1, hidden_states=None) == True

    update_module(
        module_name="transformers.models.t5.modeling_t5",
        function=transformers.models.t5.modeling_t5.T5Attention.compute_bias,
        new_function_name="updatedBias",
        modifications={"*": "return 10"},
    )
    transformers.models.t5.modeling_t5.T5Attention.compute_bias = transformers.models.t5.modeling_t5.updatedBias
    assert transformers.models.t5.modeling_t5.T5Attention.compute_bias(1, 10, 20) == 10

    # test specific modification
    src_code = (
        "vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}\n"
        "        vocab.update(self.added_tokens_encoder)\n"
    )
    update_module(
        module_name="transformers.models.t5.tokenization_t5",
        function=transformers.models.t5.tokenization_t5.T5Tokenizer.get_vocab,
        new_function_name="new_vocab",
        modifications={src_code: 'vocab = {"1": "success"}\n'},
    )
    transformers.models.t5.tokenization_t5.T5Tokenizer.get_vocab = transformers.models.t5.tokenization_t5.new_vocab
    assert transformers.models.t5.tokenization_t5.T5Tokenizer.get_vocab(1) == {"1": "success"}
