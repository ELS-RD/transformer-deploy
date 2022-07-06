# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Wrapper for ngram_repeat_block cuda extension """
from torch import nn
from torch.autograd import Function
import ngram_repeat_block_cuda

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
