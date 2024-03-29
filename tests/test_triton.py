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
import inspect
import tempfile
from pathlib import Path

import pytest
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer

from transformer_deploy.t5_utils import t5_model
from transformer_deploy.triton.configuration import EngineType
from transformer_deploy.triton.configuration_decoder import ConfigurationDec
from transformer_deploy.triton.configuration_encoder import ConfigurationEnc
from transformer_deploy.triton.configuration_question_answering import ConfigurationQuestionAnswering
from transformer_deploy.triton.configuration_t5 import ConfigurationT5Decoder, ConfigurationT5Encoder
from transformer_deploy.triton.configuration_token_classifier import ConfigurationTokenClassifier
from transformer_deploy.utils import generative_model, python_tokenizer, question_answering, token_classifier


@pytest.fixture
def working_directory() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory()


@pytest.fixture
def conf_encoder(working_directory: tempfile.TemporaryDirectory):
    conf = ConfigurationEnc(
        model_name_base="test",
        dim_output=[-1, 2],
        nb_instance=1,
        tensor_input_names=["input_ids", "attention_mask"],
        working_directory=working_directory.name,
        device="cuda",
    )
    conf.engine_type = EngineType.ONNX  # should be provided later...
    return conf


@pytest.fixture
def conf_decoder(working_directory: tempfile.TemporaryDirectory):
    conf = ConfigurationDec(
        model_name_base="test",
        dim_output=[-1, 2],
        nb_instance=1,
        tensor_input_names=["input_ids", "attention_mask"],
        working_directory=working_directory.name,
        device="cuda",
    )
    conf.engine_type = EngineType.ONNX  # should be provided later...
    return conf


@pytest.fixture
def conf_token_classifier(working_directory: tempfile.TemporaryDirectory):
    conf = ConfigurationTokenClassifier(
        model_name_base="test",
        dim_output=[-1, 2],
        nb_instance=1,
        tensor_input_names=["input_ids", "attention_mask"],
        working_directory=working_directory.name,
        device="cuda",
    )
    conf.engine_type = EngineType.ONNX
    return conf


@pytest.fixture
def conf_question_answering(working_directory: tempfile.TemporaryDirectory):
    conf = ConfigurationQuestionAnswering(
        model_name_base="test",
        dim_output=[-1, 2],
        nb_instance=1,
        tensor_input_names=["input_ids", "attention_mask"],
        working_directory=working_directory.name,
        device="cuda",
    )
    conf.engine_type = EngineType.ONNX
    return conf


@pytest.fixture
def conf_encoder_t5(working_directory: tempfile.TemporaryDirectory):
    conf = ConfigurationT5Encoder(
        model_name_base="test",
        dim_output=[-1, 2],
        nb_instance=1,
        tensor_input_names=["input_ids", "attention_mask"],
        working_directory=working_directory.name,
        device="cuda",
    )
    conf.engine_type = EngineType.ONNX  # should be provided later...
    return conf


@pytest.fixture
def conf_decoder_t5(working_directory: tempfile.TemporaryDirectory):
    conf = ConfigurationT5Decoder(
        model_name_base="test",
        dim_output=[-1, 2],
        nb_instance=1,
        tensor_input_names=["input_ids", "attention_mask"],
        working_directory=working_directory.name,
        device="cuda",
    )
    conf.engine_type = EngineType.ONNX  # should be provided later...
    return conf


def test_model_conf(conf_encoder, conf_decoder, conf_token_classifier):
    expected = """
name: "test_onnx_model"
max_batch_size: 0
platform: "onnxruntime_onnx"
default_model_filename: "model.bin"

input [
{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [-1, -1]
},
{
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [-1, -1]
}
]

output {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 2]
}

instance_group [
    {
      count: 1
      kind: KIND_GPU
    }
]
"""  # noqa: W293
    assert expected.strip() == conf_encoder.get_model_conf()
    assert expected.strip() == conf_decoder.get_model_conf()
    assert expected.strip() == conf_token_classifier.get_model_conf()


def test_tokenizer_conf(conf_encoder):
    expected = """
name: "test_onnx_tokenize"
max_batch_size: 0
backend: "python"

input [
{
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
}
]

output [
{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [-1, -1]
},
{
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [-1, -1]
}
]

instance_group [
    {
      count: 1
      kind: KIND_GPU
    }
]
"""  # noqa: W293
    assert expected.strip() == conf_encoder.get_tokenize_conf()


def test_inference_conf(conf_encoder):
    expected = """
name: "test_onnx_inference"
max_batch_size: 0
platform: "ensemble"

input [
{
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
}
]

output {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 2]
}

ensemble_scheduling {
    step [
        {
            model_name: "test_onnx_tokenize"
            model_version: -1
            input_map {
            key: "TEXT"
            value: "TEXT"
        }
        output_map [
{
    key: "input_ids"
    value: "input_ids"
},
{
    key: "attention_mask"
    value: "attention_mask"
}
        ]
        },
        {
            model_name: "test_onnx_model"
            model_version: -1
            input_map [
{
    key: "input_ids"
    value: "input_ids"
},
{
    key: "attention_mask"
    value: "attention_mask"
}
            ]
        output_map {
                key: "output"
                value: "output"
            }
        }
    ]
}
"""  # noqa: W293
    assert expected.strip() == conf_encoder.get_inference_conf()


def test_generate_conf(conf_decoder):
    expected = """
name: "test_onnx_generate"
max_batch_size: 0
backend: "python"

input [
    {
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

output [
    {
        name: "output"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

instance_group [
    {
      count: 1
      kind: KIND_GPU
    }
]

parameters: {
  key: "FORCE_CPU_ONLY_INPUT_TENSORS"
  value: {
    string_value:"no"
  }
}
"""  # noqa: W293
    print(conf_decoder.get_generation_conf())
    assert expected.strip() == conf_decoder.get_generation_conf()


def test_token_classifier_inference_conf(conf_token_classifier):
    expected = """
name: "test_onnx_inference"
max_batch_size: 0
backend: "python"

input [
    {
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

output [
    {
        name: "output"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

instance_group [
    {
      count: 1
      kind: KIND_GPU
    }
]


parameters: {
  key: "FORCE_CPU_ONLY_INPUT_TENSORS"
  value: {
    string_value:"no"
  }
}
"""
    assert expected.strip() == conf_token_classifier.get_inference_conf()


def test_question_answering_inference_conf(conf_question_answering):
    expected = """
name: "test_onnx_inference"
max_batch_size: 0
backend: "python"

input [
    {
        name: "QUESTION"
        data_type: TYPE_STRING
        dims: [ -1 ]
    },
    {
        name: "CONTEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

output [
    {
        name: "output"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

instance_group [
    {
      count: 1
      kind: KIND_GPU
    }
]


parameters: {
  key: "FORCE_CPU_ONLY_INPUT_TENSORS"
  value: {
    string_value:"no"
  }
}
"""
    assert expected.strip() == conf_question_answering.get_inference_conf()


def test_t5_encoder_inference_conf(conf_encoder_t5):
    expected = """
name: "test_onnx_inference"
max_batch_size: 0
platform: "ensemble"

input [
{
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
}
]

output {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 2]
}

ensemble_scheduling {
    step [
        {
            model_name: "test_onnx_tokenize"
            model_version: -1
            input_map {
            key: "TEXT"
            value: "TEXT"
        }
        output_map [
{
    key: "input_ids"
    value: "input_ids"
},
{
    key: "attention_mask"
    value: "attention_mask"
}
        ]
        },
        {
            model_name: "test_onnx_model"
            model_version: -1
            input_map [
{
    key: "input_ids"
    value: "input_ids"
},
{
    key: "attention_mask"
    value: "attention_mask"
}
            ]
        output_map {
                key: "output"
                value: "output"
            }
        }
    ]
}
"""
    assert expected.strip() == conf_encoder_t5.get_inference_conf()


def test_t5_decoder_generate_conf(conf_decoder_t5):
    expected = """
name: "t5_model_generate"
max_batch_size: 0
backend: "python"

input [
{
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
}
]

output {
    name: "OUTPUT_TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
}
instance_group [
    {
      count: 1
      kind: KIND_GPU
    }
]

parameters: {
    key: "FORCE_CPU_ONLY_INPUT_TENSORS"
    value: {
        string_value:"no"
    }
}
"""
    assert expected.strip() == conf_decoder_t5.get_generation_conf()


def test_create_folders(
    conf_encoder,
    conf_decoder,
    conf_token_classifier,
    conf_question_answering,
    conf_encoder_t5,
    conf_decoder_t5,
    working_directory: tempfile.TemporaryDirectory,
):
    fake_model_path = Path(working_directory.name).joinpath("fake_model.bin")
    fake_model_path.write_bytes(b"abc")

    for conf, paths, python_code in [
        (
            conf_encoder,
            [
                conf_encoder.model_folder_name,
                conf_encoder.python_folder_name,
                conf_encoder.inference_folder_name,
            ],
            python_tokenizer,
        ),
        (
            conf_decoder,
            [
                conf_decoder.model_folder_name,
                conf_decoder.python_folder_name,
                conf_decoder.inference_folder_name,
            ],
            generative_model,
        ),
        (
            conf_token_classifier,
            [
                conf_token_classifier.model_folder_name,
                conf_token_classifier.python_folder_name,
                conf_token_classifier.inference_folder_name,
            ],
            token_classifier,
        ),
        (
            conf_question_answering,
            [
                conf_question_answering.model_folder_name,
                conf_question_answering.python_folder_name,
                conf_question_answering.inference_folder_name,
            ],
            question_answering,
        ),
        (
            conf_encoder_t5,
            [
                conf_encoder_t5.model_folder_name,
                conf_encoder_t5.python_folder_name,
                conf_encoder_t5.inference_folder_name,
            ],
            t5_model,
        ),
        (
            conf_decoder_t5,
            [
                conf_decoder_t5.model_folder_name,
                conf_decoder_t5.python_folder_name,
            ],
            t5_model,
        ),
    ]:
        model_name = (
            "t5-small"
            if type(conf) in [ConfigurationT5Decoder, ConfigurationT5Encoder]
            else "philschmid/MiniLM-L6-H384-uncased-sst2"
        )
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        config: PretrainedConfig = AutoConfig.from_pretrained(model_name)
        conf.create_configs(tokenizer=tokenizer, config=config, model_path=fake_model_path, engine_type=EngineType.ONNX)
        for folder_name in paths:
            path = Path(conf.working_dir).joinpath(folder_name)
            assert path.joinpath("config.pbtxt").exists()
            assert path.joinpath("config.pbtxt").read_text() != ""
            assert path.joinpath("1").exists()

        model_path = Path(conf.working_dir).joinpath(conf.python_folder_name).joinpath("1").joinpath("model.py")
        assert model_path.exists()
        if type(conf) not in [ConfigurationT5Decoder, ConfigurationT5Encoder]:
            assert model_path.read_text() == inspect.getsource(python_code)
