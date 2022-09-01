#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Utils related to sentence-transformers.
"""

from torch import nn


try:
    # noinspection PyUnresolvedReferences
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass


class STransformerWrapper(nn.Module):
    """
    Wrap sentence-transformers model to provide a forward function with multiple inputs as expected by ONNX export tool.
    """

    def __init__(self, model: "SentenceTransformer"):
        super().__init__()
        self.model = model

    def forward(self, *kargs, **kwargs):
        inputs = dict()
        if len(kargs) >= 2:
            inputs["input_ids"] = kargs[0]
            inputs["attention_mask"] = kargs[-1]
            if len(kargs) == 3:
                inputs["token_type_ids"] = kargs[1]
        if len(kwargs) > 0:
            inputs = kwargs
        assert 2 <= len(inputs) <= 3, f"unexpected number of inputs: {len(inputs)}"
        outputs = self.model.forward(input=inputs)
        return outputs["sentence_embedding"]


def load_sentence_transformers(path: str, use_auth_token: str = None) -> STransformerWrapper:
    """
    Load sentence-transformers model and wrap it to make it behave like any other transformers model
    :param path: path to the model
    :param use_auth_token: authentication token used to access private models
    :return: wrapped sentence-transformers model
    """
    try:
        import sentence_transformers
    except ImportError:
        raise Exception(
            "sentence-transformers library is not present, you can install it using: "
            "pip install sentence-transformers==2.2.0 (or a greater version)"
        )
    from packaging import version

    assert version.parse(sentence_transformers.__version__) >= version.parse("2.2.0"), (
        f"sentence-transformers library's version is {sentence_transformers.__version__}, "
        f"you need at least the V2.2.0 version"
    )
    model: SentenceTransformer = sentence_transformers.SentenceTransformer(
        model_name_or_path=path, use_auth_token=use_auth_token
    )
    return STransformerWrapper(model=model)
