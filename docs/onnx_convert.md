# Convert Pytorch model to ONNX

To ease optimization we need to convert our Pytorch model written in imperative code in a mostly static graph.  
Therefore, optimization tooling will be able to run static analysis and search for some pattern to optimize.  
The target graph format is ONNX.

!!! quote "from https://onnx.ai/"

    ONNX is an open format built to represent machine learning models. ONNX defines a common set of operators — the building blocks of machine learning and deep learning models — and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.” (https://onnx.ai/). The format has initially been created by Facebook and Microsoft to have a bridge between Pytorch (research) and Caffee2 (production).

There are 2 ways to perform an export from Pytorch:

- `tracing mode`: send some (dummy) data to the model, and the tool will trace them inside the model, that way it will guess what the graph looks like;
- `scripting`: requires the models to be written in a certain way to work, its main advantage is that the dynamic logic is kept intact but adds many constraints in the way models are written.

!!! attention

    `Tracing mode` is not magic, for instance it can’t see operations you are doing in numpy (if any), the graph will be static, some if/else code is fixed forever, for loop will be unrolled, etc. 

:hugging: Hugging Face and model authors took care that main/most models are tracing mode compatible.

Following commented code performs the ONNX conversion:

```py linenums="1" hl_lines="17 18 19 20 30"
from collections import OrderedDict
import torch
from torch.onnx import TrainingMode

def convert_to_onnx(
    model_pytorch, output_path: str, inputs_pytorch, opset: int = 12
) -> None:
    """
    Convert a Pytorch model to an ONNX graph by tracing the provided input inside the Pytorch code.
    :param model_pytorch: Pytorch model
    :param output_path: where to save ONNX file
    :param inputs_pytorch: Tensor, can be dummy data, shape is not important as we declare all axes as dynamic.
    Should be on the same device than the model (CPU or GPU)
    :param opset: version of ONNX protocol to use, usually 12, or 13 if you use per channel quantized model
    """
    # dynamic axis == variable length axis
    dynamic_axis = OrderedDict()
    for k in inputs_pytorch.keys():
        dynamic_axis[k] = {0: "batch_size", 1: "sequence"}
    dynamic_axis["output"] = {0: "batch_size"}
    with torch.no_grad():
        torch.onnx.export(
            model_pytorch,  # model to optimize
            args=tuple(inputs_pytorch.values()),  # tuple of multiple inputs
            f=output_path,  # output path / file object
            opset_version=opset,  # the ONNX version to use, 13 if quantized model, 12 for not quantized ones
            do_constant_folding=True,  # simplify model (replace constant expressions)
            input_names=list(inputs_pytorch.keys()),  # input names
            output_names=["output"],  # output axis name
            dynamic_axes=dynamic_axis,  # declare dynamix axis for each input / output
            training=TrainingMode.EVAL,  # always put the model in evaluation mode
            verbose=False,
        )
```

!!! note

    One particular point is that we declare some axis as dynamic.  
    If we were not doing that, the graph would only accept tensors with the exact same shape that the ones we are using to build it (the dummy data), so sequence length or batch size would be fixed.  
    The name we have given to input and output fields will be reused in other tools.

A complete conversion process in real life (including TensorRT engine step) looks like that: 

![Image title](img/export_process.png)

--8<-- "resources/abbreviations.md"