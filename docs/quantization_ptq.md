# Post Training Quantization (PTQ) and Quantization Aware Training (QAT)

## What is it?

A PTQ is basically a fine tuned model where we add quantization nodes and that we calibrate.

Calibration is a key step in the static quantization process. Its quality depends on the final accuracy (the inference speed will stay the same).  
Moreover, a good PTQ is a good basis for a good Quantization Aware Training (QAT).

By calling `with QATCalibrate(...) as qat:`, the lib will patch `Transformer` model AST (source code) in RAM, basically adding quantization support to each model.

## How to tune a PTQ?

One of the things we try to guess during the calibration is what range of tensor values capture most of the information stored in the tensor.  
Indeed, a FP32 tensor can store at the same time very large and very small values, we obviously can't do the same with a 8-bits integer tensors and a scale.  
An 8-bits integer can only encode 255 values so we need to fix some limits and say, if a value is outside our limits, it just takes a maximum value instead of its real one.  
For instance, if we say our range is -1000 to +1000 and a tensor contains the value +4000, it will be replaced by the maximum value, +1000.

As said before, we will use the histogram method to find the perfect range. We also need to choose a percentile.  
Usually, you will choose something very close to 100.

!!! danger

    If the percentile is too small, we put too many values outside the covered range.  
    Values outside the range will be replaced by a single maximum value and you lose some granularity in model weights.
    
    If the percentile is too big, your range will be very large and because 8-bits signed integers can only encode values between -127 to +127, even when you use a scale you lose in granularity.

    Therefore, we in our demo, we launched a grid search on percentile hyper parameter and retrived 1 accuracy point with very little effort.

## One step further, the QAT

If it's not enough, the last step is to just fine tune a second the mode