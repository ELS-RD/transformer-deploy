# FAQ

## Is CPU deployment a viable option?

Let's start with the usual "it depends" :wink:  
It's a viable option if:

* your sequences are short or very short (<= 128 tokens)
* you use a distilled/small flavor model (like [miniLM](https://arxiv.org/abs/2002.10957) or [XtremeDistil](https://www.microsoft.com/en-us/research/uploads/prod/2020/04/XtremeDistil_ACL_2020.pdf))
* you don't use batching, as CPU are not very good at it, and using a batch size of 1 you can avoid padding
* you tune number of threads and number of inference engine instances:
using all threads available won't provide best results, 
in many cases, you will get a better throughput by using multiple instances of the inference engine. 
Triton server has an option for that.
* you use dynamic quantization

In most cases, Nvidia T4 GPU (the cheapest GPU option available on all cloud) will offer you the best perf / cost trade-off by a large margin.  
Compared to Intel Xeon of 2nd/3rd generation, they are cheaper for better results.

## What should I use to optimize GPU deployment?

* on the hardware side, 1 or more T4 GPU
* TensorRT for the optimization
* INT-8 quantization (QAT)
* fixed sequence length and dynamic batch axis

## Can I use sparse tensors if my GPU supports it?

At the time of writing, sparse tensors can only be used if you implement your model with TensorRT operators manually.  
If you import your model from ONNX, you won't see any acceleration, it should improve in Q3 2022.

## Should I use quantization or distillation or both?

... it depends, but usually quantization will bring a X2 speed up compared to an already optimized model with little cost in accuracy.  
Distillation can bring you more speed, but in many cases, will cost you more in accuracy (at least on hard NLP tasks).

## How this compares to the TensorFlow ecosystem?

Vanilla TensorFlow has a good ecosystem, it even has a basic integration of TensorRT (basic -> not all feature/optimization).   
If you need really good inference optimization, Nvidia advices in its official documentation to export the model on onnx and then follow the same optimization process than any PyTorch model.  
So, on a perf only side, there is no difference.

For sure, the idea written everywhere that for production TensorFlow is a better choice is just wrong.  
For instance, Amazon and Microsoft use Nvidia Triton inference server on most of their products using ML (like [Microsoft Office](https://reg.rainfocus.com/flow/nvidia/nvidiagtc/ap2/page/sessioncatalog/session/1629317744587001TJe7), or [advertising on Amazon](https://reg.rainfocus.com/flow/nvidia/nvidiagtc/ap2/page/sessioncatalog/session/16301005050970010fZk)), in 2021 at least.  
And Microsoft Bing [is built over TensorRT](https://blogs.bing.com/Engineering-Blog/october-2021/Bing-delivers-more-contextualized-search-using-quantized-transformer-inference-on-NVIDIA-GPUs-in-Azu).

## Should I use Torch-TensorRT?

[Torch-TensorRT](https://github.com/NVIDIA/Torch-TensorRT) is not yet mature (few ATen -> trt op support), and has not yet Triton backend (they are working on it).  
It seems that the new Pytorch Fx interface is the right direction to go from PyTorch to TensorRT.

## What do I do if I have accuracy issues with mixed precision?

Most of the time, it's an operator which overflows (too big value for FP16 numbers).  
[Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy) should help you to find the operator(s) to fix.  
Then, during engine building, you can fix the precision to FP32 for some operators.  
Check in the convert source code how we did it.

## What do I do if ONNX export fails?

Most of the time ONNX export fail because of unsupported operations.
One way to workaround is to reimplement that part or override the module with `symbolic()` static function.
More info on [https://pytorch.org/docs/stable/onnx.html#static-symbolic-method](https://pytorch.org/docs/stable/onnx.html#static-symbolic-method)

## Why don't you support GPU quantization on ONNX Runtime instead of TensorRT?

There are few reasons why ONNX Runtime GPU quantization support is not supported:

* it doesn’t support QAT, it "just" patches ONNX file and run them on TensorRT provider. Because ONNX file can’t be retrained, you can't do a fine tuning after quantization. Concretely, it means that if post training quantization doesn't provide an accuracy good enough for your use case, you won't use quantization at all.
* [QAT library from Nvidia](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) (the one used in this project) let you easily enable and disable each quantizer (on Python). You can see how useful it is in the quantization demo notebook, we used that feature to disable quantization on layernorm on specific layers to retrieve 1 point of accuracy without sacrificing perf at the post training quantization step. However, the way it currently works on ONNX Runtime is all or nothing (for each operator). Of course, if you are well verse if ONNX things, you can manually parse your graph with a graph surgery tools and make your change but it would take a lot of time compared to just for loop in your Pytorch modules.
* and more important, in our own experiment, models that contain unsupported operators by TensorRT just crashed on ONNX Runtime… It was unexpected as ONNX Runtime is supposed to split graph and leverage several providers when one doesn't support an operation. I suppose in my case that the issue is that the operator exists but not with the right type. At the end, to make it work, I needed to patch the source code. So not a better user experience.

--8<-- "resources/abbreviations.md"