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
Generate Nvidia Triton server configuration files for decoder based model (GPT-2).
"""
import inspect
from pathlib import Path
from typing import List

from transformers import PretrainedConfig, PreTrainedTokenizer

from transformer_deploy.t5_utils import t5_model
from transformer_deploy.triton.configuration import Configuration, EngineType
from transformer_deploy.utils import python_tokenizer


class ConfigurationT5Encoder(Configuration):
    @property
    def python_code(self):
        return inspect.getsource(python_tokenizer)

    @property
    def python_folder_name(self) -> str:
        return f"{self.model_name}_tokenize"

    def get_tokenize_conf(self) -> str:
        """
        Generate tokenization step configuration.
        :return: tokenization step configuration
        """
        return f"""
{self._get_header(name=self.python_folder_name, backend="python")}

input [
{{
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
}}
]

output [
{self._get_tokens()}
]

{self._instance_group()}
""".strip()

    def get_inference_conf(self) -> str:
        """
        Generate inference step configuration.
        :return: inference step configuration
        """
        output_map_blocks = list()
        for input_name in self.tensor_input_names:
            output_map_text = f"""
{{
    key: "{input_name}"
    value: "{input_name}"
}}
""".strip()
            output_map_blocks.append(output_map_text)

        mapping_keys = ",\n".join(output_map_blocks)

        return f"""
{self._get_header(name=self.inference_folder_name, platform="ensemble")}

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
    dims: {str(self.dim_output)}
}}

ensemble_scheduling {{
    step [
        {{
            model_name: "{self.python_folder_name}"
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

    def create_configs(
        self, tokenizer: PreTrainedTokenizer, config: PretrainedConfig, model_path: str, engine_type: EngineType
    ) -> None:
        super().create_configs(tokenizer=tokenizer, config=config, model_path=model_path, engine_type=engine_type)

        for path, conf_content in [
            (self.working_dir.joinpath(self.model_folder_name).joinpath("config.pbtxt"), self.get_model_conf()),
            (self.working_dir.joinpath(self.python_folder_name).joinpath("config.pbtxt"), self.get_tokenize_conf()),
            (self.working_dir.joinpath(self.inference_folder_name).joinpath("config.pbtxt"), self.get_inference_conf()),
        ]:  # type: Path, str
            path.parent.mkdir(parents=True, exist_ok=True)
            path.parent.joinpath("1").mkdir(exist_ok=True)
            path.write_text(conf_content)


class ConfigurationT5Decoder(Configuration):
    @property
    def python_code(self):
        return inspect.getsource(t5_model)

    @property
    def python_folder_name(self) -> str:
        return "t5_model_generate"

    def get_generation_conf(self) -> str:
        """
        Generate sequence configuration.
        :return: Generate sequence configuration
        """
        all_past_keys: List[str] = list()
        all_present_keys: List[str] = list()
        for i in range(self.num_layers):
            past_keys = f"""
    {{
        key: "past_key_values.{i}.decoder.key"
        value: "past_key_values.{i}.decoder.key"
    }},
    {{
        key: "past_key_values.{i}.decoder.value"
        value: "past_key_values.{i}.decoder.value"
    }},
    {{
        key: "past_key_values.{i}.encoder.key"
        value: "past_key_values.{i}.encoder.key"
    }},
    {{
        key: "past_key_values.{i}.encoder.value"
        value: "past_key_values.{i}.encoder.value"
    }}
    """
            all_past_keys.append(past_keys)
            present_keys = f"""
    {{
        key: "present.{i}.decoder.key"
        value: "present.{i}.decoder.key"
    }},
    {{
        key: "present.{i}.decoder.value"
        value: "present.{i}.decoder.value"
    }},
    {{
        key: "present.{i}.encoder.key"
        value: "present.{i}.encoder.key"
    }},
    {{
        key: "present.{i}.encoder.value"
        value: "present.{i}.encoder.value"
    }}
    """
            all_present_keys.append(present_keys)

        decoder_past_keys_inputs = ",\n".join(all_past_keys)
        decoder_present_keys_outputs = ",\n".join(all_present_keys)
        output_map_blocks = list()
        for input_name in self.tensor_input_names:
            output_map_text = f"""
    {{
        key: "{input_name}"
        value: "{input_name}"
    }}
    """.strip()
            output_map_blocks.append(output_map_text)

        return f"""
    {self._get_header(name=self.inference_folder_name, platform="ensemble")}

    input [
    {{
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }}
    ]

    output {{
        name: "OUTPUT_TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }}

    ensemble_scheduling {{
        step [
            {{
                model_name: "transformer_onnx_tokenize"
                model_version: -1
                input_map {{
                    key: "TEXT"
                    value: "TEXT"
                }}
                output_map {{
                    key: "input_ids"
                    value: "input_ids"
                }}
            }},
            {{
                model_name: "transformer_onnx_encoder"
                model_version: -1
                input_map {{
                        key: "input_ids"
                        value: "input_ids"
                    }}
                output_map {{
                    key: "output"
                    value: "encoder_hidden_states"
                }}
            }},
            {{
                model_name: "transformer_onnx_decoder"
                model_version: -1
                input_map [
                {{
                    key: "input_ids"
                    value: "input_ids"
                }},
                {{
                    key: "encoder_hidden_states"
                    value: "encoder_hidden_states"
                }},
                {{
                    key: "enable_cache"
                    value: "enable_cache"
                }},
                {decoder_past_keys_inputs}
                ]
                output_map [
                {{
                    key: "output_text"
                    value: "output_text"
                }},
                {decoder_present_keys_outputs}
                ]
            }}
        ]
    }}
    """.strip()

    def get_model_conf(self) -> str:
        """
        Generate model configuration.
        :return: model configuration
        """
        all_past_keys: List[str] = list()
        all_present_keys: List[str] = list()
        for i in range(self.num_layers):
            past_keys = f"""
        {{
            name: "past_key_values.{i}.decoder.key"
            data_type: TYPE_FP32
            dims: [-1, 8, -1, 64]
        }},
        {{
            name: "past_key_values.{i}.decoder.value"
            data_type: TYPE_FP32
            dims: [-1, 8, -1, 64]
        }},
        {{
            name: "past_key_values.{i}.encoder.key"
            data_type: TYPE_FP32
            dims: [-1, 8, -1, 64]
        }},
        {{
            name: "past_key_values.{i}.encoder.value"
            data_type: TYPE_FP32
            dims: [-1, 8, -1, 64]
        }}
        """
            all_past_keys.append(past_keys)
            present_keys = f"""
        {{
            name: "present.{i}.decoder.key"
            data_type: TYPE_FP32
            dims: [-1, -1, -1, -1]
        }},
        {{
            name: "present.{i}.decoder.value"
            data_type: TYPE_FP32
            dims: [-1, -1, -1, -1]
        }},
        {{
            name: "present.{i}.encoder.key"
            data_type: TYPE_FP32
            dims: [-1, -1, -1, -1]
        }},
        {{
            name: "present.{i}.encoder.value"
            data_type: TYPE_FP32
            dims: [-1, -1, -1, -1]
        }}
        """
            all_present_keys.append(present_keys)

        decoder_past_keys_inputs = ",\n".join(all_past_keys)
        decoder_present_keys_outputs = ",\n".join(all_present_keys)
        return f"""
name: "{self.model_folder_name}"
max_batch_size: 0
platform: "{self.inference_platform}"
default_model_filename: "model.bin"

input [
    {{
        name: "input_ids"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
    }},
    {{
        name: "encoder_hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, 512 ]
    }},
    {{
        name: "enable_cache"
        data_type: TYPE_INT32
        dims: [ 1 ]
    }},
    {decoder_past_keys_inputs}
]
output [
    {{
        name: "logits"
        data_type: TYPE_FP32
        dims: {str(self.dim_output)}
    }},
    {decoder_present_keys_outputs}
]
{self._instance_group()}
""".strip()

    def create_configs(
        self, tokenizer: PreTrainedTokenizer, config: PretrainedConfig, model_path: str, engine_type: EngineType
    ) -> None:
        super().create_configs(tokenizer=tokenizer, config=config, model_path=model_path, engine_type=engine_type)

        wd_path = Path(self.working_dir)
        for path, conf_content in [
            (wd_path.joinpath(self.model_folder_name).joinpath("config.pbtxt"), self.get_model_conf()),
            (wd_path.joinpath(self.python_folder_name).joinpath("config.pbtxt"), self.get_generation_conf()),
        ]:  # type: Path, str
            path.parent.mkdir(parents=True, exist_ok=True)
            path.parent.joinpath("1").mkdir(exist_ok=True)
            path.write_text(conf_content)
