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
from typing import Dict, List

import numpy as np


try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer, TensorType


class TritonPythonModel:
    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc

        path: str = str(Path(args["model_repository"]).parent.absolute())
        self.tokenizer = AutoTokenizer.from_pretrained(path)

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
                text=query, return_tensors=TensorType.NUMPY, padding=True, pad_to_multiple_of=8
            )
            # tensorrt uses int32 as input type, ort uses int64
            tokens_dict = {k: v.astype(np.int32) for k, v in tokens.items()}
            # communicate the tokenization results to Triton server
            outputs = list()
            for input_name in self.tokenizer.model_input_names:
                tensor_input = pb_utils.Tensor(input_name, tokens_dict[input_name])
                outputs.append(tensor_input)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses
