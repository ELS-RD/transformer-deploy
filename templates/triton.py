from enum import Enum
from pathlib import Path


class ModelType(Enum):
    ONNX = 1
    TensorRT = 2


class Configuration:
    def __init__(self, model_name: str, model_type: ModelType, batch_size: int, nb_output: int, nb_instance: int):
        self.model_name = model_name
        self.model_folder_name = f"{self.model_name}_model"
        self.tokenizer_folder_name = f"{self.model_name}_tokenize"
        self.inference_folder_name = f"{self.model_name}_inference"
        self.batch_size = batch_size
        self.nb_model_output = nb_output
        assert nb_instance > 0, f"nb_instance=={nb_instance}: nb model instances should be positive"
        self.nb_instance = nb_instance
        if model_type == ModelType.ONNX:
            self.input_type = "TYPE_INT64"
            self.platform = "onnxruntime_onnx"
        else:
            self.input_type = "TYPE_INT32"
            self.platform = "tensorrt_plan"

    def get_model_conf(self) -> str:
        # TODO manage input_type axis
        return f"""
name: "{self.model_folder_name}"
platform: "{self.platform}"
max_batch_size: {self.batch_size}

input [
    {{
        name: "input_ids"
        data_type: {self.input_type}
        dims: [-1, -1]
    }},
    {{
        name: "attention_mask"
        data_type: {self.input_type}
        dims: [-1, -1]
    }}
]

output {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, {self.nb_model_output}]
}}

instance_group [
    {{
      count: {self.nb_instance}
      kind: KIND_GPU
    }}
]
"""

    def get_tokenize_conf(self):
        # TODO manage input_type axis
        return f"""
name: "{self.tokenizer_folder_name}"
backend: "python"
max_batch_size: {self.batch_size}

input [
    {{
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }}
]

output [
    {{
        name: "INPUT_IDS"
        data_type: {self.input_type}
        dims: [ -1, -1 ]
    }},
    {{
        name: "ATTENTION"
        data_type: {self.input_type}
        dims: [ -1, -1 ]
    }}
]

instance_group [
    {{
      count: {self.nb_instance}
      kind: KIND_CPU
    }}
]
"""

    def get_inference_conf(self):
        # TODO manage input_type axis
        return f"""
name: "{self.inference_folder_name}"
platform: "ensemble"
max_batch_size: {self.batch_size}

input [
    {{
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }}
]

output {{
    name: "score"
    data_type: {self.input_type}
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
                key: "INPUT_IDS"
                value: "INPUT_IDS"
            }},
            {{
                key: "ATTENTION"
                value: "ATTENTION"
            }}
        ]
        }},
        {{
            model_name: "{self.model_name}"
            model_version: -1
            input_map [
                {{
                    key: "input_ids"
                    value: "INPUT_IDS"
                }},
                {{
                    key: "attention_mask"
                    value: "ATTENTION"
                }}
            ]
        output_map {{
                key: "output"
                value: "score"
            }}
        }}
    ]
}}
"""

    def create_folders(self, workind_directory: str):
        wd_path = Path(workind_directory)
        wd_path.mkdir(parents=True, exist_ok=True)
        for folder_name, conf_func in [(self.model_folder_name, self.get_model_conf),
                                       (self.tokenizer_folder_name, self.get_tokenize_conf),
                                       (self.inference_folder_name, self.get_inference_conf)]:
            current_folder = wd_path.joinpath(folder_name)
            current_folder.mkdir(exist_ok=False)
            conf = conf_func()
            current_folder.joinpath("config.pbtxt").write_text(conf)
            version_folder = current_folder.joinpath("1")
            version_folder.mkdir(exist_ok=False)
