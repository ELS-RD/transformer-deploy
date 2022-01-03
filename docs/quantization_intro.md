# Nvidia GPU INT-8 quantization

## What is it about?

Quantization is one of the most effective and generic approaches to make model inference faster.
Basically, it replaces high precision float numbers in model tensors encoded in 32 or 16 bits by lower precision ones encoded in 8 bits or less:

* it takes less memory
* computation is easier / faster

It can be applied to any model in theory, and, if done well, it should maintain accuracy.

The purpose of this notebook is to show a process to perform quantization on any `Transformer` architectures.

Moreover, the library is designed to offer a simple API and still let advanced users tweak the algorithm.

## Benchmark

!!! tip "TL;DR"

    We benchmarked Pytorch and Nvidia TensorRT, on both CPU and GPU, with/without quantization, our methods provide the fastest inference by large margin.

| Framework                 | Precision | Latency (ms) | Accuracy | Speedup    | Hardware |
|:--------------------------|-----------|--------------|----------|:-----------|:--------:|
| Pytorch                   | FP32      | 4267         | 86.6 %   | X 0.02     |   CPU    |
| Pytorch                   | FP16      | 4428         | 86.6 %   | X 0.02     |   CPU    |
| Pytorch                   | INT-8     | 3300         | 85.9 %   | X 0.02     |   CPU    |
| Pytorch                   | FP32      | 77           | 86.6 %   | X 1        |   GPU    |
| Pytorch                   | FP16      | 56           | 86.6 %   | X 1.38     |   GPU    |
| ONNX Runtime              | FP32      | 76           | 86.6 %   | X 1.01     |   GPU    |
| ONNX Runtime              | FP16      | 34           | 86.6 %   | X 2.26     |   GPU    |
| ONNX Runtime              | FP32      | 4023         | 86.6 %   | X 0.02     |   CPU    |
| ONNX Runtime              | FP16      | 3957         | 86.6 %   | X 0.02     |   CPU    |
| ONNX Runtime              | INT-8     | 3336         | 86.5 %   | X 0.02     |   CPU    |
| TensorRT                  | FP16      | 30           | 86.6 %   | X 2.57     |   GPU    |
| TensorRT (**our method**) | **INT-8** | **17**       | 86.2 %   | **X 4.53** | **GPU**  |


!!! note

    measures done on a Nvidia RTX 3090 GPU + 12 cores i7 Intel CPU (support AVX-2 instruction)
    Roberta `base` architecture flavor with batch of size 32 / seq len 256, similar results obtained for other sizes/seq len not included in the table.
    Accuracy obtained after a single epoch, no LR search or any hyper parameter optimization

Check the end to end demo to see where these numbers are from.
