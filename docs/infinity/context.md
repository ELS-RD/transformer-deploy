# :hugging: Hugging Face Infinity benchmark context

In sept 2021, :hugging: Hugging Face released a new product called `Infinity`.   
It’s described as a server to perform inference at *enterprise scale*.   
The communication is around the promise that the product can perform Transformer inference at 1 millisecond latency on the GPU. 
A public demo is available on YouTube :material-youtube: :

<iframe width="560" height="315" src="https://www.youtube.com/embed/jiftCAhOYQA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

According to the demo presenter, :hugging: Hugging Face Infinity server costs at least 💰20 000$/year for a single model deployed on a single machine (no information is publicly available on price scalability).

In the next parts we will try to compare this open source library with the commercial solution from :hugging: Hugging Face.
