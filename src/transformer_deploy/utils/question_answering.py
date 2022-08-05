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
This module is copy-pasted in generated Triton configuration folder to perform inference.
"""

import json

# noinspection DuplicatedCode
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from transformers.modeling_outputs import QuestionAnsweringModelOutput


try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.

from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, QuestionAnsweringPipeline


# to suppress a warning, use a known class name
# https://github.com/huggingface/transformers/blob/3f936df66287f557c6528912a9a68d7850913b9b/
# src/transformers/pipelines/base.py#L882
class BertForQuestionAnswering(PreTrainedModel):
    def __init__(self, model_name: str, model_path: str):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.model_path = model_path
        self.model_name = model_name

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids: Union[None, torch.Tensor] = None,
    ):
        input_ids = input_ids.type(dtype=torch.int32)
        attention_mask = attention_mask.type(dtype=torch.int32)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.type(dtype=torch.int32)
        inputs = [
            pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids)),
            pb_utils.Tensor.from_dlpack("attention_mask", torch.to_dlpack(attention_mask)),
        ]
        if token_type_ids is not None:
            inputs.append(pb_utils.Tensor.from_dlpack("token_type_ids", torch.to_dlpack(token_type_ids)))
        inference_request = pb_utils.InferenceRequest(
            model_name=self.model_path,
            requested_output_names=["start_logits", "end_logits"],
            inputs=inputs,
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            start_logits = pb_utils.get_output_tensor_by_name(inference_response, "start_logits")
            end_logits = pb_utils.get_output_tensor_by_name(inference_response, "end_logits")
            start_logits_tensor: torch.Tensor = torch.from_dlpack(start_logits.to_dlpack())
            end_logits_tensor: torch.Tensor = torch.from_dlpack(end_logits.to_dlpack())
            return QuestionAnsweringModelOutput(start_logits=start_logits_tensor, end_logits=end_logits_tensor)


class TritonPythonModel:
    tokenizer: PreTrainedTokenizer
    device: str
    model: BertForQuestionAnswering

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        current_name: str = str(Path(args["model_repository"]).parent.absolute())
        target_model = args["model_name"].replace("_inference", "_model")
        self.device = "cpu" if args["model_instance_kind"] == "CPU" else "cuda"
        self.model = BertForQuestionAnswering(model_name=current_name, model_path=target_model)
        self.model.to(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(current_name)
        # self.config = AutoConfig.from_pretrained(current_name)
        self.pipeline = QuestionAnsweringPipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
        )

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
            question = [
                t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "QUESTION").as_numpy().tolist()
            ]
            context = [
                t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "CONTEXT").as_numpy().tolist()
            ]

            outputs = self.pipeline(question=question[0], context=context[0])

            tensor_output = [pb_utils.Tensor("output", np.array(json.dumps(outputs), dtype=object))]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses
