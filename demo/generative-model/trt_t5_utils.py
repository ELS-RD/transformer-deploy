import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorrt as trt
import torch
from pytorch_quantization.nn import Linear
from tensorrt import Logger, Runtime
from tensorrt.tensorrt import ICudaEngine, IExecutionContext
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.file_utils import ModelOutput
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.trt_utils import get_binding_idxs, infer_tensorrt, load_engine


class T5TRTDecoder(torch.nn.Module):
    def __init__(
        self,
        trt_engine_file: str,
        config: PretrainedConfig,
        runtime: Runtime,
        device: torch.device,
        profile_index: int,
        use_cache: bool = False,
    ):
        super(T5TRTDecoder, self).__init__()
        self.main_input_name = "input_ids"  # https://github.com/huggingface/transformers/pull/14803
        self.trt_engine_file: str = trt_engine_file
        self.config: PretrainedConfig = config
        self.runtime: Runtime = runtime
        self.device = device
        self.profile_index: int = profile_index
        self.use_cache: bool = use_cache
        self.inputs: Dict[str, Optional[Tensor, bool]] = dict()
        self.timings = list()

        with open(file=self.trt_engine_file, mode="rb") as f:
            self.engine: ICudaEngine = self.runtime.deserialize_cuda_engine(f.read())
            stream: int = torch.cuda.current_stream().cuda_stream
            self.context: IExecutionContext = self.engine.create_execution_context()
            self.context.set_optimization_profile_async(profile_index=self.profile_index, stream_handle=stream)
            # retrieve input/output IDs
            self.input_binding_idxs, self.output_binding_idxs = get_binding_idxs(self.engine, profile_index)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        **_,
    ):
        self.inputs["input_ids"] = input_ids
        self.inputs["final_seq_len"] = torch.Tensor([1])
        self.inputs["encoder_hidden_states"] = encoder_hidden_states
        use_cache = self.use_cache and past_key_values is not None
        self.inputs["enable_cache"] = torch.tensor([use_cache], device=self.device, dtype=torch.bool)
        if past_key_values is not None:
            for index, (k_dec, v_dec, k_enc, v_enc) in enumerate(past_key_values):
                self.inputs[f"past_key_values.{index}.decoder.key"] = k_dec
                self.inputs[f"past_key_values.{index}.decoder.value"] = v_dec
                self.inputs[f"past_key_values.{index}.encoder.key"] = k_enc
                self.inputs[f"past_key_values.{index}.encoder.value"] = v_enc
        else:
            for i in range(self.config.num_layers):
                self.inputs[f"past_key_values.{i}.decoder.key"] = torch.zeros(
                    [input_ids.shape[0], 8, 1, 64], dtype=torch.float32
                )
                self.inputs[f"past_key_values.{i}.decoder.value"] = torch.zeros(
                    [input_ids.shape[0], 8, 1, 64], dtype=torch.float32
                )
                self.inputs[f"past_key_values.{i}.encoder.key"] = torch.zeros(
                    [input_ids.shape[0], 8, 13, 64], dtype=torch.float32
                )
                self.inputs[f"past_key_values.{i}.encoder.value"] = torch.zeros(
                    [input_ids.shape[0], 8, 13, 64], dtype=torch.float32
                )
        start_timer = time.time()
        dec_output = infer_tensorrt(
            context=self.context,
            host_inputs=self.inputs,
            input_binding_idxs=self.input_binding_idxs,
            output_binding_idxs=self.output_binding_idxs,
        )
        self.timings.append(time.time() - start_timer)
        past_states = list()
        idx_dec_out = 1
        for index in range(self.config.num_layers):
            kv = (
                dec_output[idx_dec_out],
                dec_output[idx_dec_out + 1],
                dec_output[idx_dec_out + 2],
                dec_output[idx_dec_out + 3],
            )
            idx_dec_out += 4
            past_states.append(kv)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=dec_output[0],
            past_key_values=past_states,
        )


class T5TRT(torch.nn.Module, GenerationMixin):
    def __init__(
        self,
        config: PretrainedConfig,
        encoder_engine_path: str,
        decoder_engine_path: str,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        batch_size: int = 1,
        profile_index: int = 0,
        use_cache: bool = False,
    ):
        super(T5TRT, self).__init__()
        self.main_input_name = "input_ids"
        self.config = config
        self.encoder_engine_path = encoder_engine_path
        self.decoder_engine_path = decoder_engine_path
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.profile_index = profile_index
        self.use_cache = use_cache
        trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
        self.runtime: Runtime = trt.Runtime(trt_logger)
        self.encoder_engine = load_engine(runtime=self.runtime, engine_file_path=self.encoder_engine_path)
        self.decoder_engine = (
            T5TRTDecoder(
                runtime=self.runtime,
                trt_engine_file=self.decoder_engine_path,
                config=self.config,
                profile_index=self.profile_index,
                device=self.device,
                use_cache=self.use_cache,
            )
            .cuda()
            .eval()
        )

    def get_encoder(self):
        return self.encoder_engine

    def get_decoder(self):
        return self.decoder_engine

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

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        # encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = {model_input_name: inputs_tensor}
        model_kwargs["encoder_outputs"]: ModelOutput = {"last_hidden_state": encoder(**encoder_kwargs)}

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs["last_hidden_state"].index_select(
                0, expanded_return_idx.to(encoder_outputs["last_hidden_state"].device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

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
            params["enable_cache"] = torch.tensor([False], device=self.device, dtype=torch.bool)
        else:
            params[self.main_input_name] = input_ids[:, -1:]
            params["enable_cache"] = torch.tensor([True], device=self.device, dtype=torch.bool)
            params["past_key_values"] = past

        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        **_,
    ):
        start_timer = time.monotonic()
        dec_output = self.get_decoder()(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )
        self.timings.append(time.monotonic() - start_timer)
        return Seq2SeqLMOutput(logits=dec_output.last_hidden_state, past_key_values=dec_output.past_key_values)


class ExportT5(torch.nn.Module):
    def __init__(self, decoder: T5Stack, lm_head: Linear, model_dim: int):
        super(ExportT5, self).__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.model_dim = model_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        final_seq_len: Optional[torch.Tensor],
        past_key_values: Tuple = None,
    ):
        out_dec = self.decoder.forward(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values
        )
        # Rescale output before projecting on vocab
        out_dec["last_hidden_state"] = out_dec["last_hidden_state"] * (self.model_dim ** -0.5)
        out_dec["last_hidden_state"] = self.lm_head(out_dec["last_hidden_state"])
        out_dec["past_key_values"] = list(out_dec["past_key_values"])
        for i, layer_out in enumerate(out_dec["past_key_values"]):  # type: int, Tuple
            assert len(layer_out) == 4
            layer_out_l = list(layer_out)
            for j, l in enumerate(layer_out):  # type: int, torch.Tensor
                if j <= 1:
                    layer_out_l[j] = l[:, :, : final_seq_len[0], :]
                else:
                    layer_out_l[j] = l
            out_dec["past_key_values"][i] = tuple(layer_out_l)
        out_dec["past_key_values"] = tuple(out_dec["past_key_values"])
        return out_dec


def are_equal(a: torch.Tensor, b: torch.Tensor, atol: float = 5e-1) -> None:
    assert np.allclose(a=a.detach().cpu().numpy(), b=b.detach().cpu().numpy(), atol=atol), f"{a}\n\nVS\n\n{b}"


def print_timings(name: str, total: float, inference: float):
    percent_inference = 100 * inference / total
    print(f"{name}: {total:.1f}, including inference: {inference:.1f} ({percent_inference:.1f}%)")
