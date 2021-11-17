import tempfile
from pathlib import Path

import pytest

from templates.triton import Configuration, ModelType


@pytest.fixture
def conf():
    conf = Configuration(model_name="test", model_type=ModelType.ONNX, batch_size=0, nb_output=2, nb_instance=1, include_token_type=False)
    return conf


def test_model_conf(conf: Configuration):
    expected = """
name: "test_model"
max_batch_size: 0
platform: "onnxruntime_onnx"

input [
    {
        name: "input_ids"
        data_type: TYPE_INT64
        dims: [-1, -1]
    },
    
    {
        name: "attention_mask"
        data_type: TYPE_INT64
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
"""
    assert expected.strip() == conf.get_model_conf()


def test_tokenizer_conf(conf: Configuration):
    expected = """
name: "test_model"
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
        data_type: TYPE_INT64
        dims: [-1, -1]
    },
    
    {
        name: "attention_mask"
        data_type: TYPE_INT64
        dims: [-1, -1]
    }

]

instance_group [
    {
      count: 1
      kind: KIND_GPU
    }
]
"""
    assert expected.strip() == conf.get_tokenize_conf()


def test_inference_conf(conf: Configuration):
    expected = """
name: "test_model"
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
    name: "OUTPUT"
    data_type: TYPE_INT64
    dims: [-1, 2]
}

ensemble_scheduling {
    step [
        {
            model_name: "test_tokenize"
            model_version: -1
            input_map {
            key: "TEXT"
            value: "TEXT"
        }
        output_map [
            {
                key: "INPUT_IDS"
                value: "INPUT_IDS"
            },
            
            {
                key: "ATTENTION"
                value: "ATTENTION"
            }
        ]
        },
        {
            model_name: "test"
            model_version: -1
            input_map [
                {
                    key: "input_ids"
                    value: "INPUT_IDS"
                },
                
                {
                    key: "attention_mask"
                    value: "ATTENTION"
                }
            ]
        output_map {
                key: "output"
                value: "OUTPUT"
            }
        }
    ]
}
"""
    assert expected.strip() == conf.get_inference_conf()


def test_create_folders(conf: Configuration):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        root_path = tmp_dir_name+"/models"
        conf.create_folders(root_path)
        for folder_name in [conf.model_folder_name, conf.tokenizer_folder_name, conf.inference_folder_name]:
            path = Path(root_path).joinpath(folder_name)
            assert path.joinpath("config.pbtxt").exists()
            assert path.joinpath("1").exists()
