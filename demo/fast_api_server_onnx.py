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

import numpy as np
from fastapi import FastAPI
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from transformers import AutoTokenizer, BatchEncoding, TensorType


app = FastAPI()
options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
model = InferenceSession("triton_models/model.onnx", options, providers=["CUDAExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("philschmid/MiniLM-L6-H384-uncased-sst2")


@app.get("/predict")
def predict(query: str):
    encode_dict: BatchEncoding = tokenizer(
        text=query,
        max_length=128,
        truncation=True,
        return_tensors=TensorType.NUMPY,
    )
    result: np.ndarray = model.run(None, dict(encode_dict))[0]
    return result.tolist()
