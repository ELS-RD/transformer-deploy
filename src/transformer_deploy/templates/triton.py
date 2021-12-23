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

from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, DataType, ModelOutput, ModelParameter, ModelInstanceGroup, ModelEnsembling
from google.protobuf import text_format

import logging

# Define field names
TEXT_FIELD_NM = 'TEXT'
INPUT_IDS_FIELD_NM = 'input_ids'
ATTENTION_MASK_FIELD_NM = 'attention_mask'
TOKENS_TYPE_FIELD_NM = 'token_type_ids'
LOGITS_FIELD_NM = 'output'


class ModelType(Enum):
    ONNX = 1
    TensorRT = 2

    
class Configuration:
    """
    Note: 
    Protocol Buffer template for config.pbtxt: https://raw.githubusercontent.com/triton-inference-server/common/main/protobuf/model_config.proto
    Python Protocol Buffer tutorial: https://developers.google.com/protocol-buffers/docs/pythontutorial
    """
    
    def __init__(
        self,
        working_directory: str,
        model_name: str,
        model_type: ModelType,
        batch_size: int,
        nb_output: int,  # No. of output logit fields
        nb_instance: int,
        include_token_type: bool,
        
        tokenizer_meta_dict=None,
        model_meta_dict=None
    ):
        # Model names & directories
        self.model_name = model_name
        self.model_name += "_onnx" if model_type == ModelType.ONNX else "_tensorrt"
        self.model_folder_name = f"{self.model_name}_model"
        self.tokenizer_folder_name = f"{self.model_name}_tokenize"
        self.inference_folder_name = f"{self.model_name}_inference"
        self.working_directory = working_directory
        
        self.batch_size = batch_size
        self.nb_model_output = nb_output
        assert nb_instance > 0, f"nb_instance=={nb_instance}: nb model instances should be positive"
        self.nb_instance = nb_instance
        
        self.include_token_type = include_token_type
        
        # Model metadata dicts
        self.tokenizer_meta_dict = tokenizer_meta_dict
        self.model_meta_dict = model_meta_dict
        self.inference_meta_dict = {**(tokenizer_meta_dict or {}), **(model_meta_dict or {})}
        
        # Platforms & input types
        if model_type == ModelType.ONNX:
            self.input_type = "TYPE_INT64"
            self.inference_platform = "onnxruntime_onnx"
        elif model_type == ModelType.TensorRT:
            self.input_type = "TYPE_INT32"
            self.inference_platform = "tensorrt_plan"
        else:
            raise Exception(f"unknown model type: {model_type}")
            
    def create_folders(self, tokenizer: PreTrainedTokenizer, model_path: str):
        # Create working directory
        wd_path = Path(self.working_directory)
        wd_path.mkdir(parents=True, exist_ok=True)
        
        # Generate Triton model directories and config.pbtxt for DL model, tokenizer and inference (tokenizer + model)
        for folder_name, conf_func in [
            (self.tokenizer_folder_name, self.get_tokenize_conf),
            (self.model_folder_name, self.get_model_conf),
            (self.inference_folder_name, self.get_inference_conf),
        ]:
            current_folder = wd_path.joinpath(folder_name)
            logging.info(f'Creating Triton model directory: {current_folder}')
            current_folder.mkdir(exist_ok=True)
            conf = conf_func()
            current_folder.joinpath("config.pbtxt").write_text(conf)
            version_folder = current_folder.joinpath("1")
            version_folder.mkdir(exist_ok=True)

        # -- Add tokenizer files to Triton directory --
        tokenizer_model_folder_path = wd_path.joinpath(self.tokenizer_folder_name).joinpath("1")
        tokenizer.save_pretrained(str(tokenizer_model_folder_path.absolute())) 
        
        # Update field names in python_tokenizer.py
        tokenizer_model_path = Path(__file__).absolute().parent.parent.joinpath("utils").joinpath("python_tokenizer.py")
        with open(tokenizer_model_path, 'r') as f:
            tokenizer_script = f.read()
            old_new_field_nms = [
                ["input_ids", INPUT_IDS_FIELD_NM],
                ["attention_mask", ATTENTION_MASK_FIELD_NM],
                ["token_type_ids", TOKENS_TYPE_FIELD_NM]
            ]
            for old_field_nm, new_field_nm in old_new_field_nms:
                tokenizer_script = tokenizer_script.replace(f'pb_utils.Tensor("{old_field_nm}"', f'pb_utils.Tensor("{new_field_nm}"')
                
        with open(Path(tokenizer_model_folder_path).joinpath("model.py"), 'w') as f:
            f.write(tokenizer_script)
        
        # -- Add model to Triton directory --
        model_folder_path = wd_path.joinpath(self.model_folder_name).joinpath("1")
        shutil.copy(model_path, os.path.join(model_folder_path, "model.bin"))
    
    @classmethod
    def dict_to_modelParamDict(cls, dict_i):
        if dict_i is None or len(dict_i) == 0:
            return None
        return {k: ModelParameter(string_value=str(v)) for k, v in dict_i.items()}
    
    @classmethod
    def config_to_pbtxt(cls, pb_config):
        return text_format.MessageToString(pb_config, use_short_repeated_primitives=True, use_index_order=True)
        
    def get_tokenize_conf(self):
        tokenizer_params_dict = self.dict_to_modelParamDict(self.tokenizer_meta_dict)
        
        tokenizer_config = ModelConfig(
                    name=self.tokenizer_folder_name,
                    backend="python",
                    max_batch_size=self.batch_size,
                    input=[ModelInput(name=TEXT_FIELD_NM, data_type='TYPE_STRING', dims=[-1])],
                    output=self.__get_tokens_info(ModelOutput),
                    instance_group=self.__instance_group_info(),
                    parameters=tokenizer_params_dict
                )
        tokenizer_pbtxt = self.config_to_pbtxt(tokenizer_config)
        return tokenizer_pbtxt
        
    def __get_tokens_info(self, model_IO_cls):
        '''model_IO_cls: ModelInput or ModelOutput'''
        tokens_info = [model_IO_cls(name=INPUT_IDS_FIELD_NM, data_type=self.input_type, dims=[-1, -1]),
                       model_IO_cls(name=ATTENTION_MASK_FIELD_NM, data_type=self.input_type, dims=[-1, -1])]
        
        if self.include_token_type:
            tokens_info.append(
                model_IO_cls(name=TOKENS_TYPE_FIELD_NM, data_type=triton_input_dtype, dims=[-1, -1])
            )
        return tokens_info
    
    def __instance_group_info(self):
        # Reference: https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#instance-groups
        return [ModelInstanceGroup(count=self.nb_instance, kind=ModelInstanceGroup.KIND_GPU)]

    def get_model_conf(self) -> str:
        model_params_dict = self.dict_to_modelParamDict(self.model_meta_dict)
        
        '''
        TODO Revisit:
        - data_type: Should it be inferred from tensors?
        - dims: [-1, self.nb_model_output] or [self.nb_model_output] 
        '''
        model_config = ModelConfig(
                                    name=self.model_folder_name,
                                    platform=self.inference_platform,
                                    max_batch_size=self.batch_size,
                                    default_model_filename="model.bin",
                                    input=self.__get_tokens_info(ModelInput),
                                    output=[ModelOutput(name=LOGITS_FIELD_NM, data_type='TYPE_FP32', dims=[-1, self.nb_model_output])],
                                    parameters=model_params_dict,
                                    instance_group=self.__instance_group_info()
                                )
        
        config_pbtxt = self.config_to_pbtxt(model_config)
        return config_pbtxt
        
    def get_inference_conf(self):
        # Tokenizer step config
        tokenizer_output_map = {k: k for k in [INPUT_IDS_FIELD_NM, ATTENTION_MASK_FIELD_NM] + 
                                ([TOKENS_TYPE_FIELD_NM] if self.include_token_type else [])}
        
        tokenizer_step = ModelEnsembling.Step(
                                    model_name=self.tokenizer_folder_name, 
                                    model_version=-1,
                                    input_map={TEXT_FIELD_NM: TEXT_FIELD_NM},
                                    output_map=tokenizer_output_map
                                )
        
        # Model step config
        model_steps = ModelEnsembling.Step(
                                    model_name=self.model_folder_name, 
                                    model_version=-1,
                                    input_map=tokenizer_output_map,
                                    output_map={LOGITS_FIELD_NM: LOGITS_FIELD_NM}
                                )
        
        # Ensemble config
        inference_params_dict = self.dict_to_modelParamDict(self.inference_meta_dict)
        model_config = ModelConfig(
                            name=self.inference_folder_name,
                            platform='ensemble',
                            max_batch_size=self.batch_size,
                            input=[ModelInput(name=TEXT_FIELD_NM, data_type='TYPE_STRING', dims=[-1])],
                            output=[ModelOutput(name=LOGITS_FIELD_NM, data_type='TYPE_FP32', dims=[-1, self.nb_model_output])],
                            parameters=inference_params_dict,
                            ensemble_scheduling=ModelEnsembling(step=[tokenizer_step, model_steps])
                        )
        
        # Convert to pbtxt
        config_pbtxt = self.config_to_pbtxt(model_config)
        return config_pbtxt
    
