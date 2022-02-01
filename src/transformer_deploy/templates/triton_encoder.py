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
Generate Nvidia Triton server configuration files for encoder based models (Bert, Roberta, Electra, etc.).
"""

from pathlib import Path

from transformers import PretrainedConfig, PreTrainedTokenizer

from transformer_deploy.templates.triton import ConfigurationAbs, ModelType


class Configuration(ConfigurationAbs):
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
        self, tokenizer: PreTrainedTokenizer, config: PretrainedConfig, model_path: str, model_type: ModelType
    ) -> None:
        super().create_configs(tokenizer=tokenizer, config=config, model_path=model_path, model_type=model_type)

        for path, conf_content in [
            (self.working_dir.joinpath(self.model_folder_name).joinpath("config.pbtxt"), self.get_model_conf()),
            (self.working_dir.joinpath(self.python_folder_name).joinpath("config.pbtxt"), self.get_tokenize_conf()),
            (self.working_dir.joinpath(self.inference_folder_name).joinpath("config.pbtxt"), self.get_inference_conf()),
        ]:  # type: Path, str
            path.parent.mkdir(parents=True, exist_ok=True)
            path.parent.joinpath("1").mkdir(exist_ok=True)
            path.write_text(conf_content)
