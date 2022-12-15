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

"""
This module is copy-pasted in generated Triton configuration folder to perform text generation with T5 model.
"""

import functools
import operator

# noinspection DuplicatedCode
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput


try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.

from transformers import AutoConfig, AutoTokenizer, BatchEncoding, PretrainedConfig, PreTrainedTokenizer, TensorType


class ExtT5Triton(torch.nn.Module, GenerationMixin):
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        encoder_name: str,
        decoder_name: str,
        torch_type: torch.dtype = torch.float32,
    ):
        super(ExtT5Triton, self).__init__()
        self.main_input_name = "input_ids"  # https://github.com/huggingface/transformers/pull/14803
        self.config: PretrainedConfig = config
        self.device: torch.device = device
        self.torch_type = torch_type

        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.use_cache = True
        self.timings = list()

    def encoder_onnx_inference(self, input_ids: torch.Tensor, **_) -> BaseModelOutputWithPastAndCrossAttentions:
        input_ids = input_ids.type(dtype=torch.int32)
        inputs = [pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))]
        encoder_inference_request = pb_utils.InferenceRequest(
            model_name=self.encoder_name, requested_output_names=["output"], inputs=inputs
        )
        encoder_inference_response = encoder_inference_request.exec()
        if encoder_inference_response.has_error():
            raise pb_utils.TritonModelException(encoder_inference_response.error().message())
        else:
            encoder_output = pb_utils.get_output_tensor_by_name(encoder_inference_response, "output")
            encoder_output_tnesor = torch.from_dlpack(encoder_output.to_dlpack())
            encoder_output_tnesor = encoder_output_tnesor.cuda()
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=encoder_output_tnesor.type(self.torch_type)
            )

    def decoder_onnx_inference(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        enable_cache: torch.Tensor,
        num_layers: int,
        past_key_values: Optional[torch.Tensor],
    ):
        decoder_inputs = list()
        decoder_inputs.append(
            pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(decoder_input_ids.to(torch.int32)))
        )
        decoder_inputs.append(
            pb_utils.Tensor.from_dlpack(
                "encoder_hidden_states", torch.to_dlpack(encoder_hidden_states.to(torch.float32))
            )
        )
        decoder_inputs.append(
            pb_utils.Tensor.from_dlpack("enable_cache", torch.to_dlpack(enable_cache.type(torch.int32)))
        )

        if past_key_values is not None:
            for index, (k_dec, v_dec, k_enc, v_enc) in enumerate(past_key_values):
                decoder_inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_key_values.{index}.decoder.key", torch.to_dlpack(k_dec.to(torch.float32))
                    )
                )
                decoder_inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_key_values.{index}.decoder.value", torch.to_dlpack(v_dec.to(torch.float32))
                    )
                )
                decoder_inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_key_values.{index}.encoder.key", torch.to_dlpack(k_enc.to(torch.float32))
                    )
                )
                decoder_inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_key_values.{index}.encoder.value", torch.to_dlpack(v_enc.to(torch.float32))
                    )
                )
        else:
            for index in range(num_layers):
                decoder_inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_key_values.{index}.decoder.key",
                        torch.to_dlpack(
                            torch.zeros(
                                size=(decoder_input_ids.size()[0], 8, 100, 64), dtype=torch.float32, device="cuda"
                            )
                        ),
                    )
                )
                decoder_inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_key_values.{index}.decoder.value",
                        torch.to_dlpack(
                            torch.zeros(
                                size=(decoder_input_ids.size()[0], 8, 100, 64), dtype=torch.float32, device="cuda"
                            )
                        ),
                    )
                )
                decoder_inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_key_values.{index}.encoder.key",
                        torch.to_dlpack(
                            torch.zeros(
                                size=(decoder_input_ids.size()[0], 8, 10, 64), dtype=torch.float32, device="cuda"
                            )
                        ),
                    )
                )
                decoder_inputs.append(
                    pb_utils.Tensor.from_dlpack(
                        f"past_key_values.{index}.encoder.value",
                        torch.to_dlpack(
                            torch.zeros(
                                size=(decoder_input_ids.size()[0], 8, 10, 64), dtype=torch.float32, device="cuda"
                            )
                        ),
                    )
                )

        requested_output_names = [
            [
                f"present.{index}.decoder.key",
                f"present.{index}.decoder.value",
                f"present.{index}.encoder.key",
                f"present.{index}.encoder.value",
            ]
            for index in range(num_layers)
        ]
        requested_output_names = functools.reduce(operator.iconcat, requested_output_names, ["logits"])
        decoder_inference_request = pb_utils.InferenceRequest(
            model_name=self.decoder_name, requested_output_names=requested_output_names, inputs=decoder_inputs
        )
        decoder_inference_response = decoder_inference_request.exec()
        if decoder_inference_response.has_error():
            raise pb_utils.TritonModelException(decoder_inference_response.error().message())
        else:
            output = pb_utils.get_output_tensor_by_name(decoder_inference_response, "logits")
            tensor: torch.Tensor = torch.from_dlpack(output.to_dlpack())
            logits = tensor.cuda()
            past_states = list()
            for index in range(num_layers):
                kv = (
                    torch.from_dlpack(
                        pb_utils.get_output_tensor_by_name(
                            decoder_inference_response, f"present.{index}.decoder.key"
                        ).to_dlpack()
                    ).cuda(),
                    torch.from_dlpack(
                        pb_utils.get_output_tensor_by_name(
                            decoder_inference_response, f"present.{index}.decoder.value"
                        ).to_dlpack()
                    ).cuda(),
                    torch.from_dlpack(
                        pb_utils.get_output_tensor_by_name(
                            decoder_inference_response, f"present.{index}.encoder.key"
                        ).to_dlpack()
                    ).cuda(),
                    torch.from_dlpack(
                        pb_utils.get_output_tensor_by_name(
                            decoder_inference_response, f"present.{index}.encoder.value"
                        ).to_dlpack()
                    ).cuda(),
                )
                past_states.append(kv)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=logits,
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
        if past is None:
            params[self.main_input_name] = input_ids
            params["enable_cache"] = torch.tensor([0], device="cuda", dtype=torch.int32)
        else:
            params[self.main_input_name] = input_ids[:, -1:]
            params["enable_cache"] = torch.tensor([1], device="cuda", dtype=torch.int32)
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
        dec_output = self.decoder_onnx_inference(
            decoder_input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            enable_cache=enable_cache,
            past_key_values=past_key_values,
            num_layers=self.config.num_layers,
        )
        return Seq2SeqLMOutput(logits=dec_output.last_hidden_state, past_key_values=dec_output.past_key_values)


class TritonPythonModel:
    tokenizer: PreTrainedTokenizer
    device: str

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        current_path: str = str(Path(args["model_repository"]).parent.absolute())
        self.device = "cpu" if args["model_instance_kind"] == "CPU" else "cuda"
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        model_config = AutoConfig.from_pretrained(current_path)
        self.model = ExtT5Triton(
            config=model_config,
            device=self.device,
            encoder_name="t5-encoder_onnx_model",
            decoder_name="t5-dec-if-node_onnx_model",
        )
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(current_path)
        # to silent a warning during seq generation
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().tolist()]
            tokens: BatchEncoding = self.tokenizer(
                text=query[0], return_tensors=TensorType.PYTORCH, return_attention_mask=False
            )
            # tensorrt uses int32 as input type, ort also because we force the format
            input_ids = tokens.input_ids.type(dtype=torch.int32)
            if self.device == "cuda":
                input_ids = input_ids.to("cuda")
            output_seq: torch.Tensor = self.model.generate(input_ids, min_length=10, max_length=512)
            decoded_texts: List[str] = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output_seq]
            tensor_output = [pb_utils.Tensor("OUTPUT_TEXT", np.array(t, dtype=object)) for t in decoded_texts]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses
