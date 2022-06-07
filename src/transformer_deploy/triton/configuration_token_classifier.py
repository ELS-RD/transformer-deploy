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
from transformer_deploy.utils import token_classifier
import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format

class ConfigurationTokenClassifier(Configuration):
    @property
    def python_code(self):
        return inspect.getsource(token_classifier)

    @property
    def python_folder_name(self) -> str:
        return f"{self.model_name}_inference"

    def get_inference_conf(self) -> str:
        """
        Generate inference step configuration.
        :return: inference step configuration
        """

        config = self._get_model_base(name=self.python_folder_name, backend="python")
        config.input.append(model_config.ModelInput(name="TEXT", data_type=model_config.DataType.TYPE_STRING, dims=[-1]))
        config.output.append(model_config.ModelOutput(name="output", data_type=model_config.DataType.TYPE_STRING, dims=[-1]))
        config.instance_group.append(model_config.ModelInstanceGroup(count=self.nb_instance, kind=self.device_kind))
        config.parameters["FORCE_CPU_ONLY_INPUT_TENSORS"] = model_config.ModelParameter(string_value="no")
        return text_format.MessageToString(config)

    def create_configs(
        self,
        tokenizer: PreTrainedTokenizer,
        config: PretrainedConfig,
        model_path: str,
        engine_type: EngineType,
    ) -> None:
        super().create_configs(
            tokenizer=tokenizer,
            config=config,
            model_path=model_path,
            engine_type=engine_type,
        )

        wd_path = Path(self.working_dir)
        for path, conf_content in [
            (
                wd_path.joinpath(self.model_folder_name).joinpath("config.pbtxt"),
                self.get_model_conf(),
            ),
            (
                wd_path.joinpath(self.python_folder_name).joinpath("config.pbtxt"),
                self.get_inference_conf(),
            ),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.parent.joinpath("1").mkdir(exist_ok=True)
            path.write_text(conf_content)
