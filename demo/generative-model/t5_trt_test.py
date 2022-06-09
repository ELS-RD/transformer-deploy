import time
from copy import copy
from typing import Dict, List

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import torch
from nvtx import nvtx
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, TensorType

from trt_t5_utils import T5TRT, print_timings

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids: torch.Tensor = tokenizer(
    ["translate English to French: This model is now very fast!"],
    return_tensors=TensorType.PYTORCH,
    padding=True,
).input_ids
input_ids = input_ids.to("cuda")
pytorch_model: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pytorch_model.config.use_cache = True

pytorch_model = pytorch_model.to("cuda")
t5_trt_encoder_plan = f"{model_name}-trt-enc.plan"
t5_trt_decoder_plan = f"{model_name}-trt-dec.plan"

t5_trt_wrapper = T5TRT(
    config=pytorch_model.config,
    encoder_engine_path=t5_trt_encoder_plan,
    decoder_engine_path=t5_trt_decoder_plan,
    tokenizer=tokenizer,
    device=pytorch_model.device,
    use_cache=True,
)
outputs = t5_trt_wrapper.generate(inputs=input_ids, max_length=8, num_beams=4, no_repeat_ngram_size=2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# benchmark: pytorch, onnx, tensorrt:
all_timings: Dict[str, Dict[str, List[float]]] = dict()
for seq_len, num_beam in [(100, 4), (200, 4)]:
    timings = dict()

    print(f"seq len: {seq_len} / # beam (batch size): {num_beam}")
    task = "TensorRT"
    with nvtx.annotate(
        task, color="blue"
    ):  # nvtx is for Nvidia nsight profiler, you can remove the line or install the library
        t5_trt_wrapper.set_cache(enable=False)
        # warmup
        t5_trt_wrapper.generate(inputs=input_ids, max_length=10, num_beams=num_beam, min_length=10)
        start = time.monotonic()
        t5_trt_wrapper.generate(inputs=input_ids, max_length=seq_len, num_beams=num_beam, min_length=seq_len)
        total_time = time.monotonic() - start
        print_timings(name=task, total=total_time, inference=sum(t5_trt_wrapper.timings))
        timings[f"{task}"] = t5_trt_wrapper.timings

    task = "TensorRT + cache"
    with nvtx.annotate(task, color="red"):
        t5_trt_wrapper.set_cache(enable=True)
        # warmup
        t5_trt_wrapper.generate(inputs=input_ids, max_length=10, num_beams=num_beam, min_length=10)
        start = time.monotonic()
        t5_trt_wrapper.generate(inputs=input_ids, max_length=seq_len, num_beams=num_beam, min_length=seq_len)
        total_time = time.monotonic() - start
        print_timings(name=task, total=total_time, inference=sum(t5_trt_wrapper.timings))
        timings[f"{task}"] = t5_trt_wrapper.timings

    # monckey patching of forward function to add a timer per generated token
    old_fw = pytorch_model.forward
    timing_pytorch = list()

    def new_fw(self, *args, **kwargs):
        timer_start = time.monotonic()
        res = old_fw(self, *args, **kwargs)
        torch.cuda.synchronize()  # makes timings correct without having significant impact on e2e latency
        total_time = time.monotonic() - timer_start
        timing_pytorch.append(total_time)
        return res

    task = "Pytorch"
    with nvtx.annotate(task, color="orange"):
        pytorch_model.config.use_cache = False
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                # warmup
                pytorch_model.generate(inputs=input_ids, max_length=10, num_beams=num_beam, min_length=10)
                pytorch_model.forward = new_fw.__get__(pytorch_model)
                start = time.monotonic()
                pytorch_model.generate(inputs=input_ids, max_length=seq_len, num_beams=num_beam, min_length=seq_len)
                total_time = time.monotonic() - start
                pytorch_model.forward = old_fw
                inference_time = np.sum(timing_pytorch)
                print_timings(name="Pytorch", total=total_time, inference=inference_time)
        timing_pytorch_no_cache = copy(timing_pytorch)
        timings[f"{task}"] = copy(timing_pytorch)
        timing_pytorch.clear()
    torch.cuda.empty_cache()

    task = "Pytorch + cache"
    with nvtx.annotate("Pytorch + cache", color="green"):
        pytorch_model.config.use_cache = True
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                # warmup
                pytorch_model.generate(inputs=input_ids, max_length=10, num_beams=num_beam, min_length=10)
                pytorch_model.forward = new_fw.__get__(pytorch_model)
                start = time.monotonic()
                pytorch_model.generate(inputs=input_ids, max_length=seq_len, num_beams=num_beam, min_length=seq_len)
                total_time = time.monotonic() - start
                pytorch_model.forward = old_fw
                print_timings(name="Pytorch + cache", total=total_time, inference=sum(timing_pytorch))
        timings[f"{task}"] = copy(timing_pytorch)
        timing_pytorch.clear()
    all_timings[f"{seq_len} / {num_beam}"] = timings
    torch.cuda.empty_cache()

sns.set_style("darkgrid")  # darkgrid, whitegrid, dark, white and ticks
plt.rc("axes", titlesize=15)  # fontsize of the axes title
plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
plt.rc("legend", fontsize=15)  # legend fontsize
plt.rc("font", size=13)  # controls default text sizes

colors = sns.color_palette("deep")
fig = plt.figure(constrained_layout=True, figsize=(12, 8))
subfigs = fig.subfigures(nrows=2, ncols=1)

fig.supxlabel("seq len (# tokens)")
fig.supylabel("latency (s)")
fig.suptitle(f"Small seq len and greedy search on {model_name} don't tell the whole (inference) story...")

for row, (plot_name, timings) in enumerate(all_timings.items()):
    subfigs[row].suptitle(f"setup #{1+row}: {plot_name} (seq len / beam search)")
    axs = subfigs[row].subplots(nrows=1, ncols=2)
    for col, accumulated in enumerate([False, True]):
        plot_axis = axs[col]
        for index, (k, v) in enumerate(timings.items()):
            axis = range(len(v))
            color = colors[index]
            v = np.array(v)
            # remove extreme values
            p99 = np.percentile(v, 99)
            v[v > p99] = p99
            v = np.cumsum(v) if accumulated else v
            plot_axis.scatter(axis, v, label=k, s=2)

        title = f"latency for the full sequence" if accumulated else f"latency for each token"
        plot_axis.title.set_text(title)

# legend deduplication
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc="upper left", markerscale=5)
plt.show()
