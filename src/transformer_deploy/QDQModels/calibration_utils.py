#  Copyright 2021, Lefebvre Sarrut Services
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
from typing import Optional

import torch.cuda
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from transformers import PreTrainedModel

from transformer_deploy.QDQModels.patch import add_qdq


class QATCalibrate:
    def __init__(self, method: str = "histogram", percentile: float = 99.99, per_channel: bool = True):
        assert torch.cuda.is_available(), "CUDA not available"
        self.model: Optional[PreTrainedModel] = None
        self.calib_method: str = method
        assert self.calib_method in [
            "histogram",
            "max",
            "percentile",
        ], f"unknown calibration method (for NLP): {self.calib_method}"
        self.calibration_percentile: float = percentile
        self.calibration_per_channel: bool = per_channel

    def __setup_nvidia_qat(self):
        input_desc = QuantDescriptor(num_bits=8, calib_method=self.calib_method)
        axis = (0,) if self.calibration_per_channel else None
        weight_desc = QuantDescriptor(num_bits=8, axis=axis)
        quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
        quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)

    def setup_model_qat(self, model: PreTrainedModel):
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

    def __finalize_calibration(self):
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

        self.model.cuda()

    def __enter__(self):
        add_qdq()
        self.__setup_nvidia_qat()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.__finalize_calibration()
