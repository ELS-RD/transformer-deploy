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
This module is copy-pasted in generated Triton configuration folder to perform the tokenization step.
"""

# noinspection DuplicatedCode
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.nn import Module
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.

from transformers import AutoConfig, AutoTokenizer, BatchEncoding, PretrainedConfig, PreTrainedTokenizer, TensorType


# IMPORTANT
# Some paramters are hard coded below like the sequence length to generate.
# If you want to provide some of those parameters at run time, the easiest way to proceed
# is to send to Triton server JSON (as a string), and just parse it in this module.
# Build
class GPTModelWrapper(Module, GenerationMixin):
    def __init__(
        self, config: PretrainedConfig, device: torch.device, inference: Callable[[torch.Tensor], torch.Tensor]
    ):
        super().__init__()
        self.config: PretrainedConfig = config
        self.device: torch.device = device
        self.inference: Callable[[torch.Tensor], torch.Tensor] = inference
        self.main_input_name = "input_ids"  # https://github.com/huggingface/transformers/pull/14803

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            self.main_input_name: input_ids,
        }

    def forward(self, input_ids, **_):
        logits = self.inference(input_ids)
        return CausalLMOutputWithCrossAttentions(logits=logits)


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
        target_model = args["model_name"].replace("_generate", "_model")

        def inference_triton(input_ids: torch.Tensor) -> torch.Tensor:
            input_ids = input_ids.type(dtype=torch.int32)
            inputs = [pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))]
            inference_request = pb_utils.InferenceRequest(
                model_name=target_model, requested_output_names=["output"], inputs=inputs
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(inference_response.error().message())
            else:
                output = pb_utils.get_output_tensor_by_name(inference_response, "output")
                tensor: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                tensor = tensor.cuda()
                return tensor

        self.model = GPTModelWrapper(config=model_config, device=self.device, inference=inference_triton)
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
            output_seq: torch.Tensor = self.model.generate(input_ids, max_length=32)
            decoded_texts: List[str] = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output_seq]
            tensor_output = [pb_utils.Tensor("output", np.array(t, dtype=object)) for t in decoded_texts]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses
