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
