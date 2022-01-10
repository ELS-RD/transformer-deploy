# Hugging Face Transformer submillisecond inference️ and deployment to production: 🤗 → 🤯

[![Documentation](https://img.shields.io/website?label=documentation&style=for-the-badge&up_message=online&url=https%3A%2F%2Fels-rd.github.io%2Ftransformer-deploy%2F)](https://els-rd.github.io/transformer-deploy/) [![tests](https://img.shields.io/github/workflow/status/ELS-RD/transformer-deploy/tests/main?label=tests&style=for-the-badge)](https://github.com/ELS-RD/transformer-deploy/actions/workflows/python-app.yml) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](./LICENCE) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg?style=for-the-badge)](https://www.python.org/downloads/release/python-360/)


### Optimize and deploy in **production** 🤗 Hugging Face Transformer models in a single command line.  

=> Up to 10X faster inference! <=

#### Why this tool?

<!--why-start-->

At [Lefebvre Dalloz](https://www.lefebvre-dalloz.fr/) we run in production several *semantic search engine* in the legal domain, 
in non-marketing language it's a reranker, and we based ours on `Transformer`.  
In those setup, latency is key to provide good user experience, and relevancy inference is done online for hundreds of snippets per user query.  
We have tested many solutions, and below is what we found:

[`Pytorch`](https://pytorch.org/) + [`FastAPI`](https://fastapi.tiangolo.com/) = 🐢  
Most tutorials on `Transformer` deployment in production are built over Pytorch and FastAPI.
Both are great tools but not very performant in inference.  

[`Microsoft ONNX Runtime`](https://github.com/microsoft/onnxruntime/) + [`Nvidia Triton inference server`](https://github.com/triton-inference-server/server) = ️🏃💨  
Then, if you spend some time, you can build something over ONNX Runtime and Triton inference server.
You will usually get from 2X to 4X faster inference compared to vanilla Pytorch. It's cool!  

[`Nvidia TensorRT`](https://github.com/NVIDIA/TensorRT/)  + [`Nvidia Triton inference server`](https://github.com/triton-inference-server/server) = ⚡️🏃💨💨  
However, if you want the best in class performances on GPU, there is only a single possible combination: Nvidia TensorRT and Triton.
You will usually get 5X faster inference compared to vanilla Pytorch.  
Sometimes it can raises up to **10X faster inference**.  
Buuuuttt... TensorRT can ask some efforts to master, it requires tricks not easy to come with, we implemented them for you!  

## Features

* heavily optimize transformer models for inference (CPU and GPU) -> between 5X and 10X speed-up
* deploy model on `Nvidia Triton` inference server (enterprise-grade), 6X faster than `FastAPI`
* add quantization support for both CPU and GPU
* simple to use: optimization done in a single command line!
* supported model: any model than can be exported to ONNX (-> most of them)
* supported tasks: classification, feature extraction (aka sentence-transformers dense embeddings)

<!--why-end-->

> Want to understand how it works under the hood?  
> read [🤗 Hugging Face Transformer inference UNDER 1 millisecond latency 📖](https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c?source=friends_link&sk=cd880e05c501c7880f2b9454830b8915)  
> <img src="resources/rabbit.jpg" width="120">

## Check our [documentation](https://els-rd.github.io/transformer-deploy/) for detailed instructions on how to use the package, including setup, GPU quantization support and Nvidia Triton inference server deployment.
