#  Copyright 2021, Lefebvre Sarrut Services
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

import math

import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer
from torch import nn
from transformers.activations import gelu
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

from transformer_deploy.QDQModels.utils import PatchTransformers


class QDQMultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = quant_nn.QuantLinear(in_features=config.dim, out_features=config.dim)
        self.k_lin = quant_nn.QuantLinear(in_features=config.dim, out_features=config.dim)
        self.v_lin = quant_nn.QuantLinear(in_features=config.dim, out_features=config.dim)
        self.out_lin = quant_nn.QuantLinear(in_features=config.dim, out_features=config.dim)

        self.matmul_q_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.matmul_k_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.matmul_v_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.matmul_w_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)

        scores = torch.matmul(
            self.matmul_q_input_quantizer(q), self.matmul_k_input_quantizer(k.transpose(2, 3))
        )  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(
            self.matmul_w_input_quantizer(weights), self.matmul_w_input_quantizer(v)
        )  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class QDQFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = quant_nn.QuantLinear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = quant_nn.QuantLinear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in ["relu", "gelu"], f"activation ({config.activation}) must be in ['relu', 'gelu']"
        self.activation = gelu if config.activation == "gelu" else nn.ReLU()

    def forward(self, input):
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


qdq_distilbert_mapping: PatchTransformers = PatchTransformers(
    module="transformers.models.distilbert.modeling_distilbert",
    mapping={
        "MultiHeadSelfAttention": QDQMultiHeadSelfAttention,
        "FFN": QDQFFN,
    },
)
