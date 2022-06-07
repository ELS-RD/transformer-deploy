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
import inspect
from pathlib import Path

from transformers import PretrainedConfig, PreTrainedTokenizer

from transformer_deploy.triton.configuration import Configuration, EngineType
from transformer_deploy.utils import python_tokenizer
import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format


class ConfigurationEnc(Configuration):
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
        config = self._get_model_base(name=self.python_folder_name, backend="python")
        config.input.append(model_config.ModelInput(name="TEXT", data_type=model_config.DataType.TYPE_STRING, dims=[-1]))
        config.output.extend(self._get_tokens_output())
        config.instance_group.append(model_config.ModelInstanceGroup(count=self.nb_instance, kind=self.device_kind))
        return text_format.MessageToString(config)

    def get_inference_conf(self) -> str:
        """
        Generate inference step configuration.
        :return: inference step configuration
        """
        mapping_keys = {}
        for input_name in self.tensor_input_names:
            mapping_keys[input_name] =input_name

        config = self._get_model_base(name=self.inference_folder_name, backend="ensemble")
        config.input.append(model_config.ModelInput(name="TEXT", data_type=model_config.DataType.TYPE_STRING, dims=[-1]))
        config.output.append(model_config.ModelInput(name="output", data_type=model_config.DataType.TYPE_FP32, dims=self.dim_output))
        config.ensemble_scheduling = model_config.ModelEnsembling(step=[
            model_config.ModelEnsembling.Step(
                model_name=self.python_folder_name,
                model_version=-1,
                input_map={"TEXT": "TEXT"},
                output_map= mapping_keys
            ),
            model_config.ModelEnsembling.Step(
                model_name=self.python_folder_name,
                model_version=-1,
                input_map= mapping_keys,
                output_map={"output": "output"},
            )
        ])

        return text_format.MessageToString(config)


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
