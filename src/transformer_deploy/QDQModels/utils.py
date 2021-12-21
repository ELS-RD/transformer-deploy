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
import inspect
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class PatchTransformers:
    module: str
    mapping: Dict[str, torch.nn.Module]

    def print_code(self):
        for class_name, cl in self.mapping.items():
            print("---------")
            print(class_name)
            inspect.getsource(cl)
