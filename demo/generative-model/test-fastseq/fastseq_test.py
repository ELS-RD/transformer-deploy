import time

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration

from transformer_deploy.utils.fastseq import code_patcher


def _generate(
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
    update_logits_path: bool,
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
        modification = (
            "value_states = project(\n            hidden_states, self.v, key_value_states, "
            "past_key_value[1] if past_key_value is not None else None\n        )\n"
        )
        forward_modifications = {
            modification: modification + "        if self.is_decoder and use_cache is True:\n            "
            "key_states = key_states.contiguous()\n            "
            "value_states = value_states.contiguous()\n"
        }
        code_patcher(
            module_name="transformers.models.t5.modeling_t5",
            function=transformers.models.t5.modeling_t5.T5Attention.forward,
            new_function_name="updatedForward",
            modifications=forward_modifications,
        )
        transformers.models.t5.modeling_t5.T5Attention.forward = transformers.models.t5.modeling_t5.updatedForward

        modification = (
            "reordered_layer_past_states = reordered_layer_past_states + (\n                    "
            "layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),\n                )\n"
        )
        reorder_cache_modifications = {
            "for layer_past_state in layer_past_states": "for layer_past_state in layer_past_states[0:2]",
            modification: modification + "\n            reordered_layer_past_states ="
            " (reordered_layer_past_states + layer_past_states[2:])\n",
        }
        code_patcher(
            module_name="transformers.models.t5.modeling_t5",
            function=transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache,
            new_function_name="updated_reorder_cache",
            modifications=reorder_cache_modifications,
        )
        transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache = (
            transformers.models.t5.modeling_t5.updated_reorder_cache
        )

    if update_logits_path:
        modification = (
            "for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):\n"
        )
        get_ngrams_modifications = {
            "prev_input_ids: torch.Tensor, num_hypos: int):": "prev_input_ids: torch.Tensor, num_hypos: int, pad_token_id: int = None):\n",
            modification: modification + "            if pad_token_id is None or ngram[-1] != pad_token_id:\n",
            "prev_ngram_tuple = tuple(ngram[:-1])": "    prev_ngram_tuple = tuple(ngram[:-1])",
            "generated_ngram[prev_ngram_tuple] =": "    generated_ngram[prev_ngram_tuple] ="
        }
        code_patcher(
            module_name="transformers.generation_logits_process",
            function=transformers.generation_logits_process._get_ngrams,
            new_function_name="updated_get_ngrams",
            modifications=get_ngrams_modifications,
        )
        transformers.generation_logits_process._get_ngrams = transformers.generation_logits_process.updated_get_ngrams

        get_ngrams_modifications = {
            ", num_hypos: int, cur_len: int": ", num_hypos: int, cur_len: int, pad_tokens_id: int",
            "_get_ngrams(ngram_size, prev_input_ids, num_hypos)": "updated_get_ngrams(ngram_size, prev_input_ids, num_hypos, pad_tokens_id)"
        }
        code_patcher(
            module_name="transformers.generation_logits_process",
            function=transformers.generation_logits_process._calc_banned_ngram_tokens,
            new_function_name="updated_calc_banned_ngram_tokens",
            modifications=get_ngrams_modifications,
        )
        transformers.generation_logits_process._calc_banned_ngram_tokens = transformers.generation_logits_process.updated_calc_banned_ngram_tokens

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


if __name__ == "__main__":
    source_path = "./data/cnndm_128.txt"
    expected_output_path = "./data/expected_t5_output.hypo"
    # test without transformers modifications
    modify_transformers = False
    update_logits_path = False
    transformers_modifications_test(
        modify_transformers,
        update_logits_path,
        source_path,
        expected_output_path,
        nbr_examples=120,
    )
    # test with transformers modifications
    modify_transformers = True
    update_logits_path = True
    transformers_modifications_test(
        modify_transformers,
        update_logits_path,
        source_path,
        expected_output_path,
        nbr_examples=100,
    )
    # test with transformers and logits process modifications
    modify_transformers = False
    update_logits_path = True
    transformers_modifications_test(
        modify_transformers,
        update_logits_path,
        source_path,
        expected_output_path,
        nbr_examples=100,
    )
