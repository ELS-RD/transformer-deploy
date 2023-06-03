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
Generate Nvidia Triton server configuration files.
"""

import os
import shutil
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import List, Optional

import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format
from transformers import PretrainedConfig, PreTrainedTokenizer


class EngineType(Enum):
    """
    Type of model to use
    """

    ONNX = 1
    TensorRT = 2


class Configuration(ABC):
    engine_type: Optional[EngineType]
    num_layers: Optional[int]
    vocab_size: Optional[int]
    python_code: Optional[str]

    def __init__(
        self,
        working_directory: str,
        model_name_base: str,
        dim_output: List[int],
        nb_instance: int,
        tensor_input_names: List[str],
        device: str,
    ):
        """
        Configuration file setup.
        :param working_directory: path to the working directory
        :param model_name_base: model name to use (used to call TensorRT API)
        :param dim_output: number of tensor outputs
        :param nb_instance: number of parallel instances to use. Mainly useful to optimize CPU inference.
        :param tensor_input_names: input names expected by the model
        :param device: where perform is done. One of [cpu, cuda]
        """
        self.model_name_base = model_name_base
        self.dim_output = dim_output
        assert nb_instance > 0, f"nb_instance=={nb_instance}: nb model instances should be positive"
        self.nb_instance = nb_instance
        self.tensor_input_names = tensor_input_names
        self.working_dir: Path = Path(working_directory)
        self.device_kind = (
            model_config.ModelInstanceGroup.Kind.KIND_GPU
            if device == "cuda"
            else model_config.ModelInstanceGroup.Kind.KIND_CPU
        )

    @property
    def python_folder_name(self) -> str:
        raise Exception("to implement")

    def _get_tokens(self) -> List[str]:
        """
        Generate input tensor configuration
        :return: input tensor configuration string
        """
        result: List[str] = list()
        for input_name in self.tensor_input_names:
            result.append(
                model_config.ModelInput(name=input_name, data_type=model_config.DataType.TYPE_INT32, dims=[-1, -1])
            )
        return result

    def _get_tokens_output(self) -> List[str]:
        """
        Generate input tensor configuration
        :return: input tensor configuration string
        """
        result: List[str] = list()
        for input_name in self.tensor_input_names:
            result.append(
                model_config.ModelOutput(name=input_name, data_type=model_config.DataType.TYPE_INT32, dims=[-1, -1])
            )
        return result

    @property
    def model_name(self) -> str:
        assert self.engine_type is not None
        return self.model_name_base + ("_onnx" if self.engine_type == EngineType.ONNX else "_tensorrt")

    @property
    def model_folder_name(self) -> str:
        return f"{self.model_name}_model"

    @property
    def inference_folder_name(self) -> str:
        return f"{self.model_name}_inference"

    @property
    def inference_platform(self) -> str:
        if self.engine_type == EngineType.ONNX:
            return "onnxruntime_onnx"
        elif self.engine_type == EngineType.TensorRT:
            return "tensorrt_plan"
        else:
            raise Exception(f"unknown model type: {self.engine_type}")

    @staticmethod
    def _get_model_base(name: str, platform: Optional[str] = None, backend: Optional[str] = None):
        assert platform is not None or backend is not None
        config = model_config.ModelConfig(name=name, max_batch_size=0)
        if platform is not None:
            config.platform = platform
        if backend is not None:
            config.backend = backend
        return config

    def get_model_conf(self) -> str:
        """
        Generate model configuration.
        :return: model configuration
        """
        config = model_config.ModelConfig(
            name=self.model_folder_name,
            max_batch_size=0,
            platform=self.inference_platform,
            default_model_filename="model.bin",
            input=self._get_tokens(),
            output=[
                model_config.ModelOutput(name="output", data_type=model_config.DataType.TYPE_FP32, dims=self.dim_output)
            ],
            instance_group=[model_config.ModelInstanceGroup(count=self.nb_instance, kind=self.device_kind)],
        )
        return text_format.MessageToString(config)

    def create_configs(
        self, tokenizer: PreTrainedTokenizer, config: PretrainedConfig, model_path: str, engine_type: EngineType
    ) -> None:
        """
        Create Triton configuration folder layout, generate configuration files, generate/move artefacts, etc.
        :param tokenizer: tokenizer to use
        :param config: tranformer model config to use
        :param model_path: main folder where to save configurations and artefacts
        :param engine_type: type of inference engine (ONNX or TensorRT)
        """
        try:
            self.num_layers = config.num_layers
            self.vocab_size = tokenizer.vocab_size
        except AttributeError:
            pass
        self.engine_type = engine_type
        target = self.working_dir.joinpath(self.python_folder_name).joinpath("1")
        target.mkdir(parents=True, exist_ok=True)
        target.joinpath("model.py").write_text(self.python_code)
        tokenizer.save_pretrained(str(target.absolute()))
        config.save_pretrained(str(target.absolute()))
        model_folder_path = self.working_dir.joinpath(self.model_folder_name).joinpath("1")
        model_folder_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, os.path.join(model_folder_path, "model.bin"))
