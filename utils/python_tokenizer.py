import os
from typing import Dict

import numpy as np

# noinspection PyUnresolvedReferences
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType


class TritonPythonModel:
    is_tensorrt: bool
    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]):
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        path: str = os.path.join(args["model_repository"], args["model_version"])
        model_name: str = args["model_name"]
        self.is_tensorrt = "tensorrt" in model_name
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def execute(self, requests):
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().tolist()]
            tokens: Dict[str, np.ndarray] = self.tokenizer(text=query, return_tensors=TensorType.NUMPY)
            if self.is_tensorrt:
                # tensorrt uses int32 as input type, ort uses int64
                tokens = {k: v.astype(np.int32) for k, v in tokens.items()}
            # communicate the tokenization results to Triton server
            input_ids = pb_utils.Tensor("input_ids", tokens["input_ids"])
            attention = pb_utils.Tensor("attention_mask", tokens["attention_mask"])
            outputs = [input_ids, attention]
            if "token_type_ids" in tokens.keys():
                token_type_ids = pb_utils.Tensor("token_type_ids", tokens["token_type_ids"])
                outputs.append(token_type_ids)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses
