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
import pytest


try:
    from transformer_deploy.convert import main
    from transformer_deploy.utils.args import parse_args
except ImportError:
    pass  # for CI unit tests


@pytest.mark.gpu
def test_minilm():
    commands = [
        "-m",
        "philschmid/MiniLM-L6-H384-uncased-sst2",
        "--backend",
        "tensorrt",
        "onnx",
        "-b",
        "1",
        "16",
        "16",
        "-s",
        "8",
        "8",
        "8",
    ]
    args = parse_args(commands=commands)
    main(commands=args)


@pytest.mark.gpu
def test_camembert():
    commands = [
        "-m",
        "camembert-base",
        "--backend",
        "tensorrt",
        "onnx",
        "-b",
        "1",
        "16",
        "16",
        "-s",
        "8",
        "8",
        "8",
    ]
    args = parse_args(commands=commands)
    main(commands=args)