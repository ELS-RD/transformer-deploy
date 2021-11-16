import tempfile
from pathlib import Path
from unittest import TestCase

from templates.triton import Configuration, ModelType


class TestConfiguration(TestCase):

    def setUp(self):
        self.conf = Configuration(model_name="test", model_type=ModelType.ONNX, batch_size=0, nb_output=2, nb_instance=1)

    def test_get_model_conf(self):
        expected = """
name: "test_inference"
platform: "ensemble"
max_batch_size: 0

input [
    {
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

output {
    name: "score"
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
                value: "score"
            }
        }
    ]
}
"""
        conf = self.conf.get_inference_conf()
        self.assertEqual(conf, expected)

    def test_get_tokenize_conf(self):
        expected = """
name: "test_tokenize"
backend: "python"
max_batch_size: 0

input [
    {
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

output [
    {
        name: "INPUT_IDS"
        data_type: TYPE_INT64
        dims: [ -1, -1 ]
    },
    {
        name: "ATTENTION"
        data_type: TYPE_INT64
        dims: [ -1, -1 ]
    }
]

instance_group [
    {
      count: 1
      kind: KIND_CPU
    }
]
"""
        conf = self.conf.get_tokenize_conf()
        self.assertEqual(conf, expected)

    def test_get_inference_conf(self):
        expected = """
name: "test_inference"
platform: "ensemble"
max_batch_size: 0

input [
    {
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

output {
    name: "score"
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
                value: "score"
            }
        }
    ]
}
"""
        conf = self.conf.get_inference_conf()
        self.assertEqual(conf, expected)

    def test_create_folders(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.conf.create_folders(tmp_dir_name)
            for folder_name in [self.conf.model_folder_name, self.conf.tokenizer_folder_name, self.conf.inference_folder_name]:
                path = Path(tmp_dir_name).joinpath(folder_name)
                self.assertTrue(path.joinpath("config.pbtxt").exists())
                self.assertTrue(path.joinpath("1").exists())
