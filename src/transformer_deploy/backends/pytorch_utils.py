#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Utils related to Pytorch inference.
"""

from typing import Callable, Dict

import numpy as np
import torch
from transformers import PreTrainedModel


def infer_classification_pytorch(
    model: PreTrainedModel, run_on_cuda: bool
) -> Callable[[Dict[str, torch.Tensor]], np.ndarray]:
    """
    Perform Pytorch inference for classification task
    :param model: Pytorch model (transformers)
    :param run_on_cuda: True if should be ran on GPU
    :return: a function to perform inference
    """

    def infer(inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        model_output = model(**inputs).logits.detach().cpu().numpy()  # noqa: F821
        if run_on_cuda:
            torch.cuda.synchronize()
        return model_output

    return infer


def infer_feature_extraction_pytorch(
    model: PreTrainedModel, run_on_cuda: bool
) -> Callable[[Dict[str, torch.Tensor]], np.ndarray]:
    """
    Perform Pytorch inference for feature extraction task
    :param model: Pytorch model (sentence-transformers)
    :param run_on_cuda: True if should be ran on GPU
    :return: a function to perform inference
    """

    def infer(inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        model_output = model(**inputs).detach().cpu().numpy()  # noqa: F821
        if run_on_cuda:
            torch.cuda.synchronize()
        return model_output

    return infer
