from enum import Enum
from pathlib import Path


class ModelType(Enum):
    ONNX = 1
    TensorRT = 2


class Configuration:
    def __init__(self, model_name: str, model_type: ModelType, batch_size: int, nb_output: int, nb_instance: int, include_token_type: bool):
        self.model_name = model_name
        self.model_folder_name = f"{self.model_name}_model"
        self.tokenizer_folder_name = f"{self.model_name}_tokenize"
        self.inference_folder_name = f"{self.model_name}_inference"
        self.batch_size = batch_size
        self.nb_model_output = nb_output
        assert nb_instance > 0, f"nb_instance=={nb_instance}: nb model instances should be positive"
        self.nb_instance = nb_instance
        self.include_token_type = include_token_type
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

    def create_folders(self, workind_directory: str):
        wd_path = Path(workind_directory)
        wd_path.mkdir(parents=True, exist_ok=True)
        for folder_name, conf_func in [(self.model_folder_name, self.get_model_conf),
                                       (self.tokenizer_folder_name, self.get_tokenize_conf),
                                       (self.inference_folder_name, self.get_inference_conf)]:
            current_folder = wd_path.joinpath(folder_name)
            current_folder.mkdir(exist_ok=True)
            conf = conf_func()
            current_folder.joinpath("config.pbtxt").write_text(conf)
            version_folder = current_folder.joinpath("1")
            version_folder.mkdir(exist_ok=True)
