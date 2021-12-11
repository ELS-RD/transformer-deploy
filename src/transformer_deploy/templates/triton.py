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

import os
import shutil
from enum import Enum
from pathlib import Path

from transformers import PreTrainedTokenizer


class ModelType(Enum):
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
        include_token_type: bool,
    ):
        self.model_name = model_name
        self.model_name += "_onnx" if model_type == ModelType.ONNX else "_tensorrt"
        self.model_folder_name = f"{self.model_name}_model"
        self.tokenizer_folder_name = f"{self.model_name}_tokenize"
        self.inference_folder_name = f"{self.model_name}_inference"
        self.batch_size = batch_size
        self.nb_model_output = nb_output
        assert nb_instance > 0, f"nb_instance=={nb_instance}: nb model instances should be positive"
        self.nb_instance = nb_instance
        self.include_token_type = include_token_type
        self.workind_directory = workind_directory
        if model_type == ModelType.ONNX:
            self.input_type = "TYPE_INT64"
            self.inference_platform = "onnxruntime_onnx"
        elif model_type == ModelType.TensorRT:
            self.input_type = "TYPE_INT32"
            self.inference_platform = "tensorrt_plan"
        else:
            raise Exception(f"unknown model type: {model_type}")

    def __get_tokens(self):
        token_type = ""
        if self.include_token_type:
            token_type = f"""    {{
        name: "token_type_ids"
        data_type: {self.input_type}
        dims: [-1, -1]
    }},
"""
        return f"""{{
        name: "input_ids"
        data_type: {self.input_type}
        dims: [-1, -1]
    }},
    {token_type}
    {{
        name: "attention_mask"
        data_type: {self.input_type}
        dims: [-1, -1]
    }}
"""

    def __instance_group(self):
        return f"""
instance_group [
    {{
      count: {self.nb_instance}
      kind: KIND_GPU
    }}
]
""".strip()

    def get_model_conf(self) -> str:
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
    dims: [-1, {", ".join(self.nb_model_output)}]
}}

{self.__instance_group()}
""".strip()

    def get_tokenize_conf(self):
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
        input_token_type_ids = ""
        if self.include_token_type:
            input_token_type_ids = """
            {
                key: "token_type_ids"
                value: "token_type_ids"
            },
        """.strip()
        output_token_type_ids = ""
        if self.include_token_type:
            output_token_type_ids = """
            {
                key: "token_type_ids"
                value: "token_type_ids"
            },
        """.strip()
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
    dims: [-1, {", ".join(self.nb_model_output)}]
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
            {{
                key: "input_ids"
                value: "input_ids"
            }},
            {input_token_type_ids}
            {{
                key: "attention_mask"
                value: "attention_mask"
            }}
        ]
        }},
        {{
            model_name: "{self.model_folder_name}"
            model_version: -1
            input_map [
                {{
                    key: "input_ids"
                    value: "input_ids"
                }},
                {output_token_type_ids}
                {{
                    key: "attention_mask"
                    value: "attention_mask"
                }}
            ]
        output_map {{
                key: "output"
                value: "output"
            }}
        }}
    ]
}}
""".strip()

    def create_folders(self, tokenizer: PreTrainedTokenizer, model_path: str):
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
        tokenizer_model_path = Path(__file__).absolute().parent.parent.joinpath("utils").joinpath("python_tokenizer.py")
        shutil.copy(str(tokenizer_model_path), str(Path(tokenizer_model_folder_path).joinpath("model.py")))
        model_folder_path = wd_path.joinpath(self.model_folder_name).joinpath("1")
        shutil.copy(model_path, os.path.join(model_folder_path, "model.bin"))
