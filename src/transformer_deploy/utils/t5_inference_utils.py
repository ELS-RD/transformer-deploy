import time
from typing import Dict, Optional, Tuple

import torch
from torch.nn import Linear
from transformers import PretrainedConfig
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.ort_utils import (
    create_model_for_provider,
    inference_onnx_binding,
)


class ExportT5(torch.nn.Module):
    def __init__(self, decoder: T5Stack, lm_head: Linear, model_dim):
        super(ExportT5, self).__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.model_dim = model_dim

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, past_key_values: Tuple = None):
        out_dec = self.decoder.forward(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values
        )
        # weight tying -> rescale output before projecting on vocab
        # to comment for T0 for instance
        out_dec["last_hidden_state"] = out_dec["last_hidden_state"] * (self.model_dim ** -0.5)
        out_dec["last_hidden_state"] = self.lm_head(out_dec["last_hidden_state"])
        return out_dec


# https://github.com/NVIDIA/TensorRT/blob/main/demo/HuggingFace/T5/export.py
class ExtT5(torch.nn.Module, GenerationMixin):
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        encoder_path: str,
        decoder_path: str,
        torch_type: torch.dtype = torch.float32,
    ):
        super(ExtT5, self).__init__()
        self.main_input_name = "input_ids"  # https://github.com/huggingface/transformers/pull/14803
        self.config: PretrainedConfig = config
        self.device: torch.device = device
        self.torch_type = torch_type

        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        self.encoder_onnx = create_model_for_provider(encoder_path, provider, log_severity=3)
        self.decoder_onnx = create_model_for_provider(decoder_path, provider, log_severity=3)
        self.use_cache = True
        self.timings = list()

    def encoder_onnx_inference(self, input_ids: torch.Tensor, **_) -> BaseModelOutputWithPastAndCrossAttentions:
        last_hidden_state = inference_onnx_binding(
            model_onnx=self.encoder_onnx,  # noqa: F821
            inputs={"input_ids": input_ids},
            device="cuda",
            binding=self.encoder_onnx.io_binding(),
        )["output"]
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state.type(self.torch_type))

    def decoder_onnx_inference(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        enable_cache: torch.Tensor,
        num_layers: int,
        past_key_values: Optional[torch.Tensor],
    ):
        inputs_onnx_dict = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "enable_cache": enable_cache,
        }

        if past_key_values is not None:
            for index, (k_dec, v_dec, k_enc, v_enc) in enumerate(past_key_values):
                inputs_onnx_dict[f"past_key_values.{index}.decoder.key"] = k_dec
                inputs_onnx_dict[f"past_key_values.{index}.decoder.value"] = v_dec
                inputs_onnx_dict[f"past_key_values.{index}.encoder.key"] = k_enc
                inputs_onnx_dict[f"past_key_values.{index}.encoder.value"] = v_enc

        result_dict = inference_onnx_binding(
            model_onnx=self.decoder_onnx,
            inputs=inputs_onnx_dict,
            binding=self.decoder_onnx.io_binding(),  # recycle the binding
            device="cuda",
            clone_tensor=False,  # no memory copy -> best perf and lowest memory footprint!
        )
        past_states = list()
        for index in range(num_layers):
            kv = (
                result_dict[f"present.{index}.decoder.key"],
                result_dict[f"present.{index}.decoder.value"],
                result_dict[f"present.{index}.encoder.key"],
                result_dict[f"present.{index}.encoder.value"],
            )
            past_states.append(kv)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=result_dict["logits"],
            past_key_values=past_states,
        )

    def get_encoder(self):
        return self.encoder_onnx_inference

    def get_decoder(self):
        return self.decoder_onnx_inference

    def set_cache(self, enable: bool) -> None:
        self.use_cache = enable

    # from transformers library (modeling_t5.py)
    def _reorder_cache(self, past, beam_idx):
        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def prepare_inputs_for_generation(self, input_ids, past=None, use_cache=None, **kwargs) -> Dict[str, torch.Tensor]:
        params = {
            "encoder_hidden_states": kwargs["encoder_outputs"]["last_hidden_state"],
        }
        if past is None:  # this is the 1st inferred token
            self.timings = list()
        if not self.use_cache:
            past = None
        if past is None:
            params[self.main_input_name] = input_ids
            params["enable_cache"] = torch.tensor([False], device="cuda", dtype=torch.bool)
        else:
            params[self.main_input_name] = input_ids[:, -1:]
            params["enable_cache"] = torch.tensor([True], device="cuda", dtype=torch.bool)
            params["past_key_values"] = past

        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        enable_cache: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        **_,
    ):
        start_timer = time.monotonic()
        dec_output = self.get_decoder()(
            decoder_input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            enable_cache=enable_cache,
            past_key_values=past_key_values,
            num_layers=self.config.num_layers,
        )
        self.timings.append(time.monotonic() - start_timer)
        return Seq2SeqLMOutput(logits=dec_output.last_hidden_state, past_key_values=dec_output.past_key_values)

