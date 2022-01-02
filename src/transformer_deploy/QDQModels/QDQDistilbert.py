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
This module add quantization support to all Distilbert architecture based models.
"""

from transformer_deploy.QDQModels.ast_utils import PatchModule


qdq_distilbert_mapping: PatchModule = PatchModule(
    module="transformers.models.distilbert.modeling_distilbert",
)
