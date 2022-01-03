# Add quantization support to any model

The idea is to take the source code of a specific model and add automatically `QDQ` nodes. 

QDQ nodes will be placed before and after an operation that we want to quantize, that’s inside these nodes that the information to perform the mapping between high precision and low precision number is stored.

That way, quantization will work out of the box for the final user.

The process is based on Python AST modification, basically we parse the model source code in RAM, we convert it to a tree, then we patch the tree to add the QDQ nodes and we replace, still in RAM, the original module source code. Our library also offer the option to restore original behavior.

In theory it works for any model. However, not related to quantization, some models are not fully compliant with `TensorRT` (unsupported operators, etc.).
For those models, we rewrite some part of the source code, these patches are manually written but are applied to the model at run time (like the AST manipulation).

??? info

    concrete examples on `Roberta` architecture: in :hugging: Hugging Face library, 
    there is a `cumsum` operator used during the position embedding generation. 
    Something very simple. It takes as input an integer tensor and output an integer tensor. 
    It happens that the `cumsum` operator from TensorRT supports float but not integer 
    (https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md). 
    It leads to a crash during the model conversion with a strange error message. 
    Converting the input to float tensor fixes the issue. 

The process is simple:

* Calibrate, after that you have a PTQ
* second fine tuning, it's the QAT (optional)

??? info

    there are many ways to get a QDQ model, you can modify Pytorch source code (including doing it at runtime like here), patch ONNX graph (this approach is used at Microsoft for instance but only support PTQ, not QAT as ONNX file can't be trained on Pytorch for now) or leverage the new FX Pytorch interface (it's a bit experimental and it seems to miss some feature to support Nvidia QAT library). Modifying the source code is the most straightforward, and doing it through AST is the least intrusive (no need to duplicate the work of HF).

Concretly, minimal user code looks like that:

```python title="apply_quantization.py" linenums="1" hl_lines="7 10"
import torch
from transformers import AutoModelForSequenceClassification
from transformer_deploy.QDQModels.calibration_utils import QATCalibrate

my_data_loader = ...

with QATCalibrate() as qat:
    model_q = AutoModelForSequenceClassification.from_pretrained("my/model")
    model_q.cuda()
    qat.setup_model_qat(model_q)
    with torch.no_grad():
        for data in my_data_loader:
            model_q(**data)  # <- calibration happens here

# export ONNX and perform inference...
```

!!! info

    The first `context manager` will enable quantization support, and the `setup_model_qat()` method will add the QDQ nodes.