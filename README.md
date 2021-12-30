# Hugging Face Transformer submillisecond inference️ and deployment to production: 🤗 → 🤯

[![tests](https://github.com/ELS-RD/transformer-deploy/actions/workflows/python-app.yml/badge.svg)](https://github.com/ELS-RD/transformer-deploy/actions/workflows/python-app.yml) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENCE) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

### Optimize and deploy in **production** Hugging Face Transformer models in a single command line.  

=> Up to 10X faster inference! <=


## 🤔 Why this tool?

[`Pytorch`](https://pytorch.org/) + [`FastAPI`](https://fastapi.tiangolo.com/) = 🐢  
Most tutorials on transformer deployment in production are built over Pytorch and FastAPI.
Both are great tools but not very performant in inference.  

[`Microsoft ONNX Runtime`](https://github.com/microsoft/onnxruntime/) + [`Nvidia Triton inference server`](https://github.com/triton-inference-server/server) = ️🏃💨  
Then, if you spend some time, you can build something over ONNX Runtime and Triton inference server.
You will usually get from 2X to 4X faster inference compared to vanilla Pytorch. It's cool!  

[`Nvidia TensorRT`](https://github.com/NVIDIA/TensorRT/)  + [`Nvidia Triton inference server`](https://github.com/triton-inference-server/server) = ⚡️🏃💨💨  
However, if you want the best in class performances on GPU, there is only a single possible combination: Nvidia TensorRT and Triton.
You will usually get 5X faster inference compared to vanilla Pytorch. 
Sometimes it can raises up to **10X faster inference**.
Buuuuttt... TensorRT is not easy to use, even less with Transformer models, it requires specific tricks not easy to come with.  


> Want to understand how it works under the hood?  
> read [📕 Hugging Face Transformer inference UNDER 1 millisecond latency 📖](https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c?source=friends_link&sk=cd880e05c501c7880f2b9454830b8915)  
> <img src="resources/rabbit.jpg" width="120">

