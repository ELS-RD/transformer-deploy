import importlib
import inspect
import time

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration


def update_transformers():
    # update forward function in T5Attention: ["transformers.models.t5.modeling_t5.T5Attention.forward()"]
    model_module = importlib.import_module(name="transformers.models.t5.modeling_t5")
    original_t5_attention = inspect.getsource(transformers.models.t5.modeling_t5.T5Attention.forward)
    code_to_find = (
        "value_states = project(\n            hidden_states, self.v, key_value_states, "
        "past_key_value[1] if past_key_value is not None else None\n        )\n"
    )
    code_to_add = (
        "        if self.is_decoder and use_cache is True:\n            "
        "key_states = key_states.contiguous()\n            "
        "value_states = value_states.contiguous()\n"
    )
    modeified_attention_t5 = original_t5_attention.replace(code_to_find, code_to_find + code_to_add)
    modeified_attention_t5 = modeified_attention_t5.replace("def forward", "def updatedForward")
    exec(inspect.cleandoc(modeified_attention_t5), model_module.__dict__, model_module.__dict__)
    transformers.models.t5.modeling_t5.T5Attention.forward = transformers.models.t5.modeling_t5.updatedForward

    # update forward function in T5Attention:
    # ["transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache()"]
    original_t5_generation = inspect.getsource(
        transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache
    )
    modified_t5_generation = original_t5_generation.replace(
        "self, past, beam_idx):", "\n        self,\n        past,\n        beam_idx,\n    ):"
    )
    code_to_find = "for layer_past_state in layer_past_states"
    code_to_add = "[0:2]"
    modified_t5_generation = modified_t5_generation.replace(code_to_find, code_to_find + code_to_add)
    code_to_find = (
        "reordered_layer_past_states = reordered_layer_past_states + (\n                    "
        "layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),\n                )\n"
    )
    code_to_add = "\n            reordered_layer_past_states = (reordered_layer_past_states + layer_past_states[2:])\n"
    modified_t5_generation = modified_t5_generation.replace(code_to_find, code_to_find + code_to_add)
    modified_t5_generation = modified_t5_generation.replace("def _reorder_cache", "def updated_reorder_cache")
    exec(inspect.cleandoc(modified_t5_generation), model_module.__dict__, model_module.__dict__)
    transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache = (
        transformers.models.t5.modeling_t5.updated_reorder_cache
    )

    return


def _generate(
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
    model_name = "t5-base"
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
                _generate(
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
                _generate(
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
                _generate(
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

        end = time.time()
        samples_per_second = processed_sample_count / (end - start)
        print(
            f"Finish the processing of {processed_sample_count} samples with the speed {samples_per_second:.2f} "
            f"samples/second. Full time: {end - start} "
        )

    for i, output in enumerate(outputs):
        if output != expected_outputs[i]:
            pass


if __name__ == "__main__":
    source_path = "./data/cnndm_128.txt"
    expected_output_path = "./data/expected_t5_output.hypo"
    # test with transformers modifications
    modify_transformers = False
    transformers_modifications_test(
        modify_transformers,
        source_path,
        expected_output_path,
        nbr_examples=40,
    )
    # test without transformers modifications
    modify_transformers = True
    transformers_modifications_test(
        modify_transformers,
        source_path,
        expected_output_path,
        nbr_examples=40,
    )
