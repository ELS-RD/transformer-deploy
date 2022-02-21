# High-level comparison

## Inference engine

The inference engine performs the computation only, it doesn't manage the communication part (HTTP/GRPC API, etc.). 

!!! summary

    * don't use Pytorch in production for inference
    * ONNX Runtime is your good enough API for most inference jobs
    * if you need best performances, use TensorRT

|                                                           | Nvidia TensorRT                                         | :material-microsoft: Microsoft ONNX Runtime                | :material-facebook: Meta Pytorch              | comments                                                                                                                        |
|:----------------------------------------------------------|:--------------------------------------------------------|:-----------------------------------------------------------|:----------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|
| :octicons-rocket-16: transformer-deploy support           | :material-check:                                        | :material-check:                                           | :material-cancel:                             |                                                                                                                                 |
| :material-license: Licence                                | Apache 2, optimization engine is closed source          | MIT                                                        | Modified BSD                                  |                                                                                                                                 |
| :material-api: ease of use (API)                          | :fontawesome-regular-angry:                             | :material-check-all:                                       | :material-check-all:                          | Nvidia has chosen to not hide technical details + model is specific to a single `hardware + model + data shapes` association    |
| :material-file-document-edit: ease of use (documentation) | :material-spider-thread: <br/> (spread out, incomplete) | :material-check: <br/> (improving)                         | :material-check-all: <br/> (strong community) |                                                                                                                                 |
| :octicons-cpu-16: Hardware support                        | :material-check: <br/> GPU + Jetson                     | :material-check-all: <br/> CPU + GPU + IoT + Edge + Mobile | :material-check: <br/> CPU + GPU              |                                                                                                                                 |
| :octicons-stopwatch-16: Performance                       | :material-speedometer:                                  | :material-speedometer-medium:                              | :material-speedometer-slow:                   | TensorRT is usually 5 to 10X faster than Pytorch when you use quantization, etc.                                                | 
| :material-target: Accuracy                                | :material-speedometer-medium:                           | :material-speedometer:                                     | :material-speedometer:                        | TensorRT optimizations may be a bit too aggressive and decrease model accuracy. It requires manual modification to retrieve it. |

## Inference HTTP/GRPC server

|                                                           | Nvidia Triton          | :material-facebook: Meta TorchServe | FastAPI                     | comments                                                                               |
|:----------------------------------------------------------|:-----------------------|:------------------------------------|:----------------------------|:---------------------------------------------------------------------------------------|
| :octicons-rocket-16: transformer-deploy support           | :material-check:       | :material-cancel:                   | :material-cancel:           |                                                                                        |
| :material-license: Licence                                | Modified BSD           | Apache 2                            | MIT                         |                                                                                        |
| :material-api: ease of use (API)                          | :material-check:       | :material-check:                    | :material-check-all:        | As a classic HTTP server, FastAPI may appear easier to use                             |
| :material-file-document-edit: ease of use (documentation) | :material-check:       | :material-check:                    | :material-check-all:        | FastAPI has one of the most beautiful documentation ever!                              |
| :octicons-stopwatch-16:  Performance                      | :material-speedometer: | :material-speedometer-medium:       | :material-speedometer-slow: | FastAPI is 6-10X slower to manage user query than Triton                               |
| **Support**                                               |                        |                                     |                             |                                                                                        |
| :octicons-cpu-16: CPU                                     | :material-check:       | :material-check:                    | :material-check:            |                                                                                        |
| :octicons-cpu-16: GPU                                     | :material-check:       | :material-check:                    | :material-check:            |                                                                                        |
| dynamic batching                                          | :material-check:       | :material-check:                    | :material-cancel:           | combine individual inference requests together to improve inference throughput         |
| concurrent model execution                                | :material-check:       | :material-check:                    | :material-cancel:           | run multiple models (or multiple instances of the same model)                          |
| pipeline                                                  | :material-check:       | :material-cancel:                   | :material-cancel:           | one or more models and the connection of input and output tensors between those models |
| native multiple backends* support                         | :material-check:       | :material-cancel:                   | :material-check:            | *backends: Microsoft ONNX Runtime, Nvidia Triton, Meta Pytorch                         |
| REST API                                                  | :material-check:       | :material-check:                    | :material-check:            |                                                                                        |
| GRPC API                                                  | :material-check:       | :material-check:                    | :material-cancel:           |                                                                                        |
| Inference metrics                                         | :material-check:       | :material-check:                    | :material-cancel:           | GPU utilization, server throughput, and server latency                                 |



--8<-- "resources/abbreviations.md"