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
Setup and run quantization calibration
"""

from typing import Optional

import torch.cuda
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from transformers import PreTrainedModel

from transformer_deploy.QDQModels.patch import add_qdq, remove_qdq


class QATCalibrate:
    def __init__(self, method: str = "histogram", percentile: float = 99.999, per_channel: bool = True):
        """
        Calibration will learn how a float tensor should be mapped to an integer tensor.
        Will learn range, bias and scale.
        Quantization targets signe 8 bits integers as it's the best supported type for Nvidia GPUs
        (there are dedicated 8 bits integer tensor cores on most modern Nvidia GPU architectures).
        Don't forget to call setup_model_qat at some point.
        :param method: the method calibration to use. One of [histogram, percentile].
        Recommended method for transformers is "histogram".
        :param percentile: for histogram method, what do you define as an outlier value
        :param per_channel: calibration granularity. per channel == per dimension.
        """
        assert torch.cuda.is_available(), "CUDA not available"
        self.model: Optional[PreTrainedModel] = None
        assert method in [
            "histogram",
            "max",
        ], f"unknown calibration method (for NLP): {method}"
        self.calib_method: str = method
        self.calibration_percentile: float = percentile
        self.calibration_per_channel: bool = per_channel

    def setup_nvidia_qat(self) -> None:
        """
        Setup Nvidia QAT library global variables.
        Should be called before initializing a model.
        """
        input_desc = QuantDescriptor(num_bits=8, calib_method=self.calib_method)
        axis = (0,) if self.calibration_per_channel else None
        weight_desc = QuantDescriptor(num_bits=8, axis=axis)
        quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
        quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)

    def setup_model_qat(self, model: PreTrainedModel) -> None:
        """
        Enable calibration on each tensor to quantize.
        :param model: model to optimize
        """
        self.model = model
        model = self.model.cuda()
        # Find the TensorQuantizer and enable calibration
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

    def finalize_calibration(self) -> None:
        """
        Disable calibration process and enable quantized nodes.
        """
        calib_method = "max" if self.calib_method == "max" else "percentile"
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        # strict=False -> avoid Exception when some quantizer are never used
                        # (because of a condition for instance)
                        module.load_calib_amax(calib_method, percentile=self.calibration_percentile, strict=False)
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
        # move back model to GPU memory
        self.model.cuda()

    @staticmethod
    def restore():
        """
        Restore behavior without quantization support.
        """
        remove_qdq()

    def __enter__(self):
        add_qdq()
        self.setup_nvidia_qat()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finalize_calibration()
