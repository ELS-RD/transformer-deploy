# Demo

In the subfolders, you will find some experiments that we hope you will find interesting.

## Infinity

This code is related to the article [Hugging Face Transformer inference UNDER 1 millisecond latency](https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c?source=friends_link&sk=cd880e05c501c7880f2b9454830b8915).
It shows how with only open source tools you can easily get better performances than commercial solution from Hugging Face.  
You will get inference in the millisecond range on a cheap T4 GPU (the cheapest option from AWS).

It includes end to end code to reproduce benchmarks published in the Medium article linked above.

## Quantization

A notebook explaining end to end how to apply GPU quantization to a transformer model.
It also includes code to significantly improve accuracy by disabling quantization on sensitive nodes.
Whith this technic expect X4-X5 faster inference than vanilla Pytorch.

## Generative model

Decoder based model like `GPT-2` have similar architecture than Bert but are definitly different beast.
In the notebook we show how IO is important.
At the end, we get X4 speedup compared to Hugging Face code.

## Question answering

Example of a question answering model server request using triton.
A notebook explaining how to create [query_body.bin](question-answering/query_body.bin) for a question answering model.
for cURL request.

## TorchDynamo

`TorchDynamo` is a promising system to get the speedup of a model compiler and the flexibility of Pytorch.  
In this experiment we benchmark the tools with more traditional approaches.
