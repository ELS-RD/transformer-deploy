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

import inspect
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import List

from transformers import PreTrainedTokenizer

from transformer_deploy.utils import python_tokenizer


class ModelType(Enum):
    """
    Type of model to use
    """

    ONNX = 1
    TensorRT = 2


class Configuration:
    def __init__(
        self,
        workind_directory: str,
        model_name: str,
        model_type: ModelType,
        batch_size: int,
        nb_output: int,
        nb_instance: int,
        input_names: List[str],
        device: str,
    ):
        """
        Configuration file setup.
        :param workind_directory: path to the working directory
        :param model_name: model name to use (used to call TensorRT API)
        :param model_type: type of model (ONNX or TensorRT)
        :param batch_size: dynamic batch size to use (0 to disable)
        :param nb_output: number of tensor outputs
        :param nb_instance: number of parallel instances to use. Mainly useful to optimize CPU inference.
        :param input_names: input names expected by the model
        :param device: where perform is done. One of [cpu, cuda]
        """
        self.model_name = model_name
        self.model_name += "_onnx" if model_type == ModelType.ONNX else "_tensorrt"
        self.model_folder_name = f"{self.model_name}_model"
        self.tokenizer_folder_name = f"{self.model_name}_tokenize"
        self.inference_folder_name = f"{self.model_name}_inference"
        self.batch_size = batch_size
        self.nb_model_output = nb_output
        assert nb_instance > 0, f"nb_instance=={nb_instance}: nb model instances should be positive"
        self.nb_instance = nb_instance
        self.input_names = input_names
        self.workind_directory = workind_directory
        if model_type == ModelType.ONNX:
            self.inference_platform = "onnxruntime_onnx"
        elif model_type == ModelType.TensorRT:
            self.inference_platform = "tensorrt_plan"
        else:
            raise Exception(f"unknown model type: {model_type}")
        self.device_kind = "KIND_GPU" if device == "cuda" else "KIND_CPU"

    def __get_tokens(self):
        """
        Generate input tensor configuration
        :return: input tensor configuration string
        """
        result: List[str] = list()
        for input_name in self.input_names:
            text = f"""
{{
    name: "{input_name}"
    data_type: TYPE_INT32
    dims: [-1, -1]
}}
""".strip()
            result.append(text)
        return ",\n".join(result)

    def __instance_group(self):
        """
        Generate instance configuration.
        :return: instance configuration
        """
        return f"""
instance_group [
    {{
      count: {self.nb_instance}
      kind: {self.device_kind}
    }}
]
""".strip()

    def get_model_conf(self) -> str:
        """
        Generate model configuration.
        :return: model configuration
        """
        return f"""
name: "{self.model_folder_name}"
max_batch_size: {self.batch_size}
platform: "{self.inference_platform}"
default_model_filename: "model.bin"

input [
{self.__get_tokens()}
]

output {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, {self.nb_model_output}]
}}

{self.__instance_group()}
""".strip()

    def get_tokenize_conf(self):
        """
        Generate tokenization step configuration.
        :return: tokenization step configuration
        """
        return f"""
name: "{self.tokenizer_folder_name}"
max_batch_size: {self.batch_size}
backend: "python"

input [
{{
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
}}
]

output [
{self.__get_tokens()}
]

{self.__instance_group()}
""".strip()

    def get_inference_conf(self):
        """
        Generate inference step configuration.
        :return: inference step configuration
        """
        output_map_blocks = list()
        for input_name in self.input_names:
            output_map_text = f"""
{{
    key: "{input_name}"
    value: "{input_name}"
}}
""".strip()
            output_map_blocks.append(output_map_text)

        mapping_keys = ",\n".join(output_map_blocks)

        return f"""
name: "{self.inference_folder_name}"
max_batch_size: {self.batch_size}
platform: "ensemble"

input [
{{
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
}}
]

output {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, {self.nb_model_output}]
}}

ensemble_scheduling {{
    step [
        {{
            model_name: "{self.tokenizer_folder_name}"
            model_version: -1
            input_map {{
            key: "TEXT"
            value: "TEXT"
        }}
        output_map [
{mapping_keys}
        ]
        }},
        {{
            model_name: "{self.model_folder_name}"
            model_version: -1
            input_map [
{mapping_keys}
            ]
        output_map {{
                key: "output"
                value: "output"
            }}
        }}
    ]
}}
""".strip()

    def create_folders(self, tokenizer: PreTrainedTokenizer, model_path: str) -> None:
        """
        Generate configuration folder layout.
        :param tokenizer: tokenizer to use
        :param model_path: ouput path
        """
        wd_path = Path(self.workind_directory)
        wd_path.mkdir(parents=True, exist_ok=True)
        for folder_name, conf_func in [
            (self.model_folder_name, self.get_model_conf),
            (self.tokenizer_folder_name, self.get_tokenize_conf),
            (self.inference_folder_name, self.get_inference_conf),
        ]:
            current_folder = wd_path.joinpath(folder_name)
            current_folder.mkdir(exist_ok=True)
            conf = conf_func()
            current_folder.joinpath("config.pbtxt").write_text(conf)
            version_folder = current_folder.joinpath("1")
            version_folder.mkdir(exist_ok=True)

        tokenizer_model_folder_path = wd_path.joinpath(self.tokenizer_folder_name).joinpath("1")
        tokenizer.save_pretrained(str(tokenizer_model_folder_path.absolute()))
        source_code: str = inspect.getsource(python_tokenizer)
        Path(tokenizer_model_folder_path).joinpath("model.py").write_text(source_code)
        model_folder_path = wd_path.joinpath(self.model_folder_name).joinpath("1")
        shutil.copy(model_path, os.path.join(model_folder_path, "model.bin"))
