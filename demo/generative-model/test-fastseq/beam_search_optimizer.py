# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for beam search related parts in Transformers."""

import logging
import warnings
from collections import UserDict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from fastseq.logging import get_logger
from fastseq.utils.api_decorator import replace
from torch import nn
from transformers.file_utils import ModelOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    _calc_banned_ngram_tokens,
    _get_generated_ngrams,
    _get_ngrams,
)
from transformers.generation_stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation_utils import (
    BeamSampleOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchOutput,
    GenerationMixin,
    GreedySearchOutput,
    SampleOutput,
    StoppingCriteriaList,
)
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModel, GPT2LMHeadModel, GPT2Model
from transformers.models.prophetnet.modeling_prophetnet import ProphetNetForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from torch import nn
from torch.autograd import Function
import ngram_repeat_block_cuda


logger = get_logger(__name__, logging.INFO)


class NGramRepeatBlockFunction(Function):
    """
    forward inputs to ngram_repeat_block cuda extension
    backward method not needed.

    """
    def forward(self, tokens, lprobs, bsz,
        step, beam_size, no_repeat_ngram_size):
        """
        Args:
        tokens(Tensor): Input tokens(Bsz*beam, seq_len)
        lprobs(Tensor): likelihood probability
        Expected to be updated in place.(Bsz*beam, vocab_size)
        bsz(int): batch size
        step(int): current step
        beam_size(int): beam size
        no_repeat_ngram_size(int): Ngram size
        """
        outputs = ngram_repeat_block_cuda.forward(tokens,
        lprobs, bsz, step, beam_size, no_repeat_ngram_size)
        return outputs

    def backward (*args):
        raise NotImplementedError

class NGramRepeatBlock(nn.Module):
    """ Wrapper class for calling ngram_repeat_block cuda extension """
    def __init__(self):
        super(NGramRepeatBlock, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, tokens, lprobs, bsz,
        step, beam_size, no_repeat_ngram_size):
        """
        Args:
        tokens(Tensor): Input tokens(Bsz*beam, seq_len)
        lprobs(Tensor): likelihood probability,
        Expected to be updated in place.(Bsz*beam, vocab_size)
        bsz(int): batch size
        step(int): current step
        beam_size(int): beam size
        no_repeat_ngram_size(int): Ngram size
        """
        assert tokens.size(0) == bsz*beam_size
        assert lprobs.size(0) == bsz*beam_size
        tokens = tokens.contiguous()
        lprobs = lprobs.contiguous()
        return NGramRepeatBlockFunction.apply(tokens, lprobs,
               bsz, step, beam_size, no_repeat_ngram_size)


no_repeat_ngram_op = NGramRepeatBlock()


@replace(_get_ngrams)
def _get_ngrams_v2(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, pad_token_id: int = None
) -> List[Dict]:
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            if pad_token_id is None or ngram[-1] != pad_token_id:
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


@replace(_calc_banned_ngram_tokens)
def _calc_banned_ngram_tokens_v2(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int, pad_tokens_id: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = _get_ngrams_v2(ngram_size, prev_input_ids, num_hypos, pad_tokens_id)

    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


@replace(NoRepeatNGramLogitsProcessor)
class NoRepeatNGramLogitsProcessorV2(NoRepeatNGramLogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that enforces no repetition of n-grams. See `Fairseq
    <https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345>`__.
    Args:
        ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int, num_beams: int = None, batch_size: int = None, pad_token_id: int = None):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        def _update_scores(banned_tokens):
            banned_idx = [
                (bbsz_idx, banned_idx)
                for bbsz_idx in range(len(banned_tokens))
                for banned_idx in banned_tokens[bbsz_idx]
            ]
            if banned_idx:
                banned_2d_idx = tuple(torch.LongTensor(list(zip(*banned_idx))))
                scores.index_put_(banned_2d_idx, scores.new_tensor([-float("inf") * banned_2d_idx[0].nelement()]))

        cpu_input_ids = input_ids.cpu()
        cur_len = input_ids.shape[-1]
        if input_ids.is_cuda and scores.is_cuda:
            if self.num_beams is not None and self.batch_size is not None:
                x = self.num_beams * self.batch_size
                if x == input_ids.size(0) and x == scores.size(0):
                    scores = no_repeat_ngram_op(
                        input_ids, scores.float(), self.batch_size, cur_len - 1, self.num_beams, self.ngram_size
                    )
                    return scores

        num_batch_hypotheses = scores.shape[0]
        banned_batch_tokens = _calc_banned_ngram_tokens_v2(
            self.ngram_size, cpu_input_ids, num_batch_hypotheses, cur_len, self.pad_token_id
        )
        _update_scores(banned_batch_tokens)
        return scores


@replace(GenerationMixin)
class GenerationMixinV2(GenerationMixin):
    """
    A class contraining all of the functions supporting generation, to be used
    as a mixin in PreTrainedModel.
    """

    def _update_beam_size(self, num_beams):
        """
        Update num_beams in the decoder's self_attn and encoder_decoder_attn
        layers if they have been optimized.
        Different implementations of ConditionalGeneration class (e.g.
        T5ForConditionalGeneration and BartForConditionalGeneration) may have
        different attribute hierarchies and their self_attn and
        encoder_decoder_attn may have been optimized or not. As a result, this
        function need to handle different cases without breaking the program.
        """

        # Update num_beams for BART decoder attention layer
        if isinstance(self, BartForConditionalGeneration):
            for layer in self.model.decoder.layers:
                layer.encoder_attn.num_beams = num_beams
                layer.self_attn.num_beams = num_beams
            logger.debug("num_beams has been updated to {}".format(num_beams))
            return

        # Update num_beams for T5 decoder attention layer
        if isinstance(self, T5ForConditionalGeneration):
            for block in self.decoder.block:
                block.layer[0].SelfAttention.num_beams = num_beams
                block.layer[1].EncDecAttention.num_beams = num_beams
            logger.debug("num_beams has been updated to {}".format(num_beams))
            return

        if isinstance(self, GPT2Model):
            for block in self.h:
                block.attn.num_beams = num_beams
            logger.debug("num_beams has been updated to {}".format(num_beams))
            return

        if isinstance(self, GPT2LMHeadModel) or isinstance(self, GPT2DoubleHeadsModel):
            for block in self.transformer.h:
                block.attn.num_beams = num_beams
            logger.debug("num_beams has been updated to {}".format(num_beams))
            return

        if isinstance(self, ProphetNetForConditionalGeneration):
            for layer in self.prophetnet.decoder.layers:
                layer.cross_attn.num_beams = num_beams
            logger.debug("num_beams has been updated to {}".format(num_beams))
            return

        logger.debug(
            "The num_beams optimization in self_attn and encoder_decoder_attn "
            "does not support {} yet.".format(self.__class__)
        )

    def _get_logits_processor(
        self,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        encoder_input_ids: torch.LongTensor,
        bad_words_ids: List[List[int]],
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        remove_invalid_values: bool,
        batch_size: int,
    ) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
        """
        processors = LogitsProcessorList()

        # init warp parameters
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        encoder_no_repeat_ngram_size = (
            encoder_no_repeat_ngram_size
            if encoder_no_repeat_ngram_size is not None
            else self.config.encoder_no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        min_length = min_length if min_length is not None else self.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        diversity_penalty = diversity_penalty if diversity_penalty is not None else self.config.diversity_penalty
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )
        # instantiate processors list

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if diversity_penalty is not None and diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(
                NoRepeatNGramLogitsProcessorV2(no_repeat_ngram_size, num_beams, batch_size, self.config.pad_token_id)
            )
        if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
            if self.config.is_encoder_decoder:
                processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
            else:
                raise ValueError("It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture")
        if bad_words_ids is not None:
            processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        if remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        return processors

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.
        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.
        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.
        Parameters:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it with
                :obj:`bos_token_id` and a batch size of 1.
            max_length (:obj:`int`, `optional`, defaults to :obj:`model.config.max_length`):
                The maximum length of the sequence to be generated.
            max_new_tokens (:obj:`int`, `optional`, defaults to None):
                The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
                :obj:`max_new_tokens` or :obj:`max_length` but not both, they serve the same purpose.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the
                ``decoder_input_ids``.
            bad_words_ids(:obj:`List[List[int]]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer(bad_word,
                add_prefix_space=True).input_ids`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            max_time(:obj:`float`, `optional`, defaults to None):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
                shape as :obj:`input_ids` that masks the pad token. `What are attention masks?
                <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
                beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
            diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
                enabled.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID :obj:`batch_id` and
                :obj:`input_ids`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the batch ID :obj:`batch_id` and the previously generated tokens :obj:`inputs_ids`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            forced_bos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the first generated token after the :obj:`decoder_start_token_id`.
                Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token
                needs to be the target language token.
            forced_eos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the last generated token when :obj:`max_length` is reached.
            remove_invalid_values (:obj:`bool`, `optional`):
                Whether to remove possible `nan` and `inf` outputs of the model to prevent the generation method to
                crash. Note that using ``remove_invalid_values`` can slow down generation.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If the
                model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific
                kwargs should be prefixed with `decoder_`.
        Return:
            :class:`~transformers.file_utils.ModelOutput` or :obj:`torch.LongTensor`: A
            :class:`~transformers.file_utils.ModelOutput` (if ``return_dict_in_generate=True`` or when
            ``config.return_dict_in_generate=True``) or a :obj:`torch.FloatTensor`.
                If the model is `not` an encoder-decoder model (``model.config.is_encoder_decoder=False``), the
                possible :class:`~transformers.file_utils.ModelOutput` types are:
                    - :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput`
                If the model is an encoder-decoder model (``model.config.is_encoder_decoder=True``), the possible
                :class:`~transformers.file_utils.ModelOutput` types are:
                    - :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.SampleEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput`
        Examples::
            >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> # do greedy decoding without providing a prompt
            >>> outputs = model.generate(max_length=40)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            >>> document = (
            ... "at least two people were killed in a suspected bomb attack on a passenger bus "
            ... "in the strife-torn southern philippines on monday , the military said."
            ... )
            >>> # encode input context
            >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
            >>> # generate 3 independent sequences using beam search decoding (5 beams)
            >>> # with T5 encoder-decoder model conditioned on short news article.
            >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> input_context = "The dog"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate 3 candidates using sampling
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
            >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
            >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
            >>> # "Legal" is one of the control codes for ctrl
            >>> input_context = "Legal My neighbor is"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> input_context = "My cute dog"
            >>> # get tokens of words that should not be generated
            >>> bad_words_ids = [tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in ["idiot", "stupid", "shut up"]]
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate sequences without allowing bad_words to be generated
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        """

        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # update beam size
        if is_beam_gen_mode:
            self._update_beam_size(num_beams)

        # Storing encoder_input_ids for logits_processor that could use them
        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
                )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = (
                max_new_tokens + input_ids.shape[-1]
                if input_ids is not None
                else max_length + model_kwargs["inputs_embeds"].shape[1]
            )
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )

        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            batch_size=input_ids.shape[0],
        )

        stopping_criteria = self._get_stopping_criteria(max_length=max_length, max_time=max_time)

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # get probability distribution warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            beam_scorer = BeamSearchScorerV2(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            batch_size = input_ids.shape[0] * num_return_sequences

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # interleave with `num_beams * num_return_sequences`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            diverse_beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.group_beam_search(
                input_ids,
                diverse_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.
        Parameters:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (:obj:`BeamScorer`):
                An derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.
            max_length (:obj:`int`, `optional`, defaults to 20):
                **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.
        Return:
            :class:`~transformers.generation_utilsBeamSearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.
        Examples::
            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForSeq2SeqLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    BeamSearchScorer,
            ... )
            >>> import torch
            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            >>> encoder_input_str = "translate English to German: How old are you?"
            >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
            >>> # lets run beam search using 3 beams
            >>> num_beams = 3
            >>> # define decoder start token ids
            >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
            >>> input_ids = input_ids * model.config.decoder_start_token_id
            >>> # add encoder_outputs to model keyword arguments
            >>> model_kwargs = {
            ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            ... }
            >>> # instantiate beam scorer
            >>> beam_scorer = BeamSearchScorer(
            ...     batch_size=1,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ... )
            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ... ])
            >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens // vocab_size).long()
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids, next_token_scores, next_tokens, next_indices, eos_token_id=eos_token_id
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


class BeamSearchScorerV2(BeamSearchScorer):
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:

        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        beams_offset = (torch.arange(0, batch_size) * self.num_beams).unsqueeze(1).type_as(input_ids)
        cand_size = 2 * self.num_beams
        cand_offsets = torch.arange(0, cand_size).type_as(input_ids)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        effective_beam_id = next_indices + beams_offset
        if eos_token_id is not None:
            eos_mask = next_tokens.eq(eos_token_id)
        else:
            eos_mask = torch.zeros_like(next_tokens).bool()
        eos_effective_idx = torch.masked_select(
            effective_beam_id[:, : self.num_beams], mask=eos_mask[:, : self.num_beams]
        )

        eos_effective_scores = torch.masked_select(next_scores[:, : self.num_beams], mask=eos_mask[:, : self.num_beams])
        input_ids_cpu = input_ids.cpu()
        eos_effective_idx_cpu = eos_effective_idx.cpu()
        eos_effective_scores_cpu = eos_effective_scores.cpu()
        for i in range(0, eos_effective_idx_cpu.size()[-1]):
            batch_idx = eos_effective_idx_cpu[i] // self.num_beams
            if not self._done[batch_idx]:
                self._beam_hyps[batch_idx.item()].add(
                    input_ids_cpu[eos_effective_idx_cpu[i]].clone(), eos_effective_scores_cpu[i]
                )
            self._done[batch_idx] = self._done[batch_idx] or self._beam_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        active_mask = torch.add(eos_mask.type_as(cand_offsets) * cand_size, cand_offsets[: eos_mask.size(1)])
        _, active_hypos = torch.topk(active_mask, k=self.num_beams, dim=1, largest=False)
        active_effective_beam_id = torch.gather(effective_beam_id, dim=1, index=active_hypos)
        active_scores = torch.gather(next_scores, dim=1, index=active_hypos)
        active_tokens = torch.gather(next_tokens, dim=1, index=active_hypos)
        beam_idxs = active_effective_beam_id.view(-1)
        beam_scores = active_scores.view(-1)
        beam_tokens = active_tokens.view(-1)

        return UserDict(
            {"next_beam_scores": beam_scores, "next_beam_tokens": beam_tokens, "next_beam_indices": beam_idxs}
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        batch_size = len(self._done)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )
