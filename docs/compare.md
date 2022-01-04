# High level comparaison

!!! summary

    * don't use Pytorch in production for inference
    * ONNX Runtime is your good enough API for most inference jobs
    * if you need best performances, use TensorRT

|                                                           | Nvidia TensorRT                                         | :material-microsoft: Microsoft ONNX Runtime                | :material-facebook: Meta Pytorch              | comments                                                                                                                        |
|:----------------------------------------------------------|:--------------------------------------------------------|:-----------------------------------------------------------|:----------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|
| :material-license: Licence                                | Apache 2, optimization engine is closed source          | MIT                                                        | Modified BSD                                  |                                                                                                                                 |
| :material-api: ease of use (API)                          | :fontawesome-regular-angry:                             | :material-check-all:                                       | ::material-check-all:                         | Nvidia has chosen to not hide technical details + model is specific to a single `hardware + model + data shapes` association    |
| :material-file-document-edit: ease of use (documentation) | :material-spider-thread: <br/> (spread out, incomplete) | :material-check: <br/> (improving)                         | :material-check-all: <br/> (strong community) |                                                                                                                                 |
| :octicons-cpu-16: Hardware support                        | :material-check: <br/> GPU + Jetson                     | :material-check-all: <br/> CPU + GPU + IoT + Edge + Mobile | :material-check: <br/> CPU + GPU              |                                                                                                                                 |
| :octicons-stopwatch-16: Performance                       | :material-speedometer:                                  | :material-speedometer-medium:                              | :material-speedometer-slow:                   | TensorRT is usually 5 to 10X faster than Pytorch when you use quantization, etc.                                                | 
| :material-target: Accuracy                                | :material-speedometer-medium:                           | :material-speedometer:                                     | :material-speedometer:                        | TensorRT optimizations may be a bit too aggressive and decrease model accuracy. It requires manual modification to retrieve it. |

--8<-- "resources/abbreviations.md"