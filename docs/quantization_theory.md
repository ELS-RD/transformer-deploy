# Some theory

## A (very) short intro to INT-8 quantization

Basic idea behind model quantization is to replace tensors made of float numbers (usually encoded on 32 bits) by lower precision representation (integers encoded on 8 bits for Nvidia GPUs).
Therefore computation is faster and model memory footprint is lower. Making tensor storage smaller makes memory transfer faster... and is also a source of computation acceleration.
This approach is very interesting for its trade-off: you reduce inference time significantly, and it costs close to nothing in accuracy.

Replacing float numbers by integers is done through a mapping.
This step is called `calibration`, and its purpose is to compute for each tensor or each channel of a tensor (one of its dimensions) a range covering most weights and then define a scale and a distribution center to map float numbers to 8 bits integers.

There are several ways to perform quantization, depending of how and when the `calibration` is performed:

* dynamically: the mapping is done online, during the inference, there are some overhead but it's usually the easiest to leverage, end user has very few configuration to set,
* statically, after training (`post training quantization` or `PTQ`): this way is efficient because quantization is done offline, before inference, but it may have an accuracy cost,
* statically, after training (`quantization aware training` or `QAT`): like a PTQ followed by a second fine tuning. Same efficiency but usually slightly better accuracy.

Nvidia GPUs don't support dynamic quantization, CPU supports all types of quantization.  
Compared to `PTQ`, `QAT` better preserves accuracy and should be preferred in most cases.

During the *quantization aware training*:

* in the inside, Pytorch will train with high precision float numbers,
* on the outside, Pytorch will simulate that a quantization has already been applied and output results accordingly (for loss computation for instance)

The simulation process is done through the add of quantization / dequantization nodes, most often called `QDQ`, it's an abbreviation you will see often in the quantization world.

!!! info "Want to learn more about quantization?"

    * You can check this [high quality blog post](https://leimao.github.io/article/Neural-Networks-Quantization/) for more information.
    * The process is well described in this [Nvidia presentation](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)

## Why does it matter?

CPU quantization is supported out of the box by `Pytorch` and `ONNX Runtime`.
**GPU quantization on the other side requires specific tools and process to be applied**.

In the specific case of `Transformer` models, few demos from Nvidia and Microsoft exist; they are all for the old vanilla Bert architecture.

It doesn't support modern architectures out of the box, like `Albert`, `Roberta`, `Deberta` or `Electra`.