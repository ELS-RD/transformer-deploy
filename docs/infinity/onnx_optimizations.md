# Offline graph optimization

As said previously, some optimizations are applied just after loading the model in memory. There is also the possibility to apply some deeper optimizations while performing a static analysis of the graph, so that it will be easier to manage the dynamics axis or drop some unnecessary casting nodes. Moreover, changing model precision (from FP32 to FP16) requires being offline. Check this guide to learn more about those optimizations.

ONNX Runtime offers such things in its tools folder. Most classical transformer architectures are supported, and it includes miniLM. You can run the optimizations through the command line:

```shell
python -m onnxruntime.transformers.optimizer ...
```

In our case, we will perform them in Python code to have a single command to execute. In the code below, we enable all possible optimizations plus perform a conversion to float 16 precision.

```python
# ONNX Runtime offline optimization code (Image by Author)
```

A part of the performance improvement comes from some approximations performed at the CUDA level: on the activation layer (GELU) and on the attention mask layer. Those approximations can have a small impact on the model outputs. In my experience, it has less effect on the model accuracy than using a different seed during training.

Regarding TensorRT, there are no offline optimizations but it’s advised in ONNX Runtime documentation to perform symbolic shape inference on the vanilla ONNX model. This is useful because it may happen that ONNX Runtime splits the graph in several subgraphs, and because of that the (tensor) shape information is lost for TensorRT. Symbolic shape inference will put back the information everywhere it’s required. And if like me, you are wondering why it’s called symbolic, it’s because it will really perform symbolic computation with a Python lib called sympy dedicated to… symbolic computation in general.