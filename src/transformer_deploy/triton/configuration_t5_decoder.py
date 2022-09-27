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

from transformer_deploy.triton.configuration import Configuration, EngineType
from transformer_deploy.utils import generative_model


class ConfigurationT5Decoder(Configuration):
    @property
    def python_code(self):
        return inspect.getsource(generative_model)

    @property
    def python_folder_name(self) -> str:
        return f"{self.model_name}_generate"

    def get_generation_conf(self) -> str:
        """
        Generate sequence configuration.
        :return: Generate sequence configuration
        """
        result: List[str] = list()
        for i in range(self.num_layers):
            text = f"""
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
            result.append(text)

        decoder_past_keys_inputs = ",\n".join(result)
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
        name: "logits"
        data_type: TYPE_FP32
        dims: {str(self.dim_output)}
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
                output_map {{
                    key: "logits"
                    value: "logits"
                }}
            }}
        ]
    }}
    """.strip()

    def get_model_conf(self) -> str:
        """
        Generate model configuration.
        :return: model configuration
        """
        result: List[str] = list()
        for i in range(self.num_layers):
            text = f"""
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
            result.append(text)

        decoder_past_keys_inputs = ",\n".join(result)
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
        data_type: TYPE_BOOL
        dims: [ 1 ]
    }},
    {decoder_past_keys_inputs}
]
output [
    {{
        name: "logits"
        data_type: TYPE_FP32
        dims: [ -1, -1, {self.vocab_size} ]
    }}
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
