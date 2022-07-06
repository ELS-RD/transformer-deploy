# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for T5 model"""

import logging
import torch
from torch import nn

from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
from transformers.models.t5.modeling_t5 import T5Attention, T5ForConditionalGeneration

from fastseq.logging import get_logger
from fastseq.utils.api_decorator import replace

logger = get_logger(__name__, logging.INFO)

@replace(T5Attention)
class T5AttentionV2(T5Attention):
    """Optimized T5Attention for self-attn and encoder-decoder-attn in T5."""

    def __init__(self,
                 config: T5Config,
                 has_relative_attention_bias=False,
                 num_beams=1):
        super().__init__(
            config=config,
            has_relative_attention_bias=has_relative_attention_bias)
        self.num_beams = num_beams

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        is_encoder_decoder_attn = key_value_states is not None

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        if self.is_decoder and use_cache is True:
            if is_encoder_decoder_attn:
                if past_key_value is None:
                    key_states = key_states.view(batch_size // self.num_beams, self.num_beams,
                               self.n_heads, key_length,
                               self.key_value_proj_dim)[:, 0:1, :, :, :].contiguous()
                    value_states = value_states.view(batch_size // self.num_beams, self.num_beams,
                               self.n_heads, key_length,
                               self.key_value_proj_dim)[:, 0:1, :, :, :].contiguous()
            present_key_value_state = (key_states, value_states)
        else:
            present_key_value_state = None

        if is_encoder_decoder_attn and use_cache:
            new_query_states = query_states.view(batch_size // self.num_beams, self.num_beams, self.n_heads,
                           seq_length, self.key_value_proj_dim)
            scores = torch.einsum(
                "bmnqd,bxnkd->bmnqk", new_query_states, key_states).reshape(
                    -1, self.n_heads, seq_length, key_length)  # (bs, n_heads, qlen, klen)
        else:
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
            # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores = scores + position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        if is_encoder_decoder_attn and use_cache:
            tmp_weights = attn_weights.view(batch_size // self.num_beams, self.num_beams,
                                       self.n_heads, seq_length, key_length)
            attn_output = torch.einsum(
                "bmnqk,bxnkd->bmnqd", tmp_weights, value_states).reshape(
                    -1, self.n_heads, seq_length, self.key_value_proj_dim
                    )  # (bs, n_heads, qlen, dim_per_head)
        else:
            attn_output = torch.matmul(attn_weights, value_states)  # (bs, n_heads, qlen, dim_per_head)
        attn_output = unshape(attn_output)  # (bs, qlen, dim)

        attn_output = self.o(attn_output)

        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

@replace(T5ForConditionalGeneration)
class T5ForConditionalGenerationV2(T5ForConditionalGeneration):
    """Optimized T5ForConditionalGenerationV2"""

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states[0:2]:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            reordered_layer_past_states = (reordered_layer_past_states + layer_past_states[2:])

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[T5Config] = T5ForConditionalGenerationV2  # pylint: disable=line-too-long