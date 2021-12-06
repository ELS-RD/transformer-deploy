import logging

import numpy as np
import pytorch_quantization.nn as quant_nn
import torch
from datasets import load_dataset, load_metric
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    IntervalStrategy,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from transformer_deploy.backends.ort_utils import convert_to_onnx
from transformer_deploy.QDQModels.QDQRoberta import QDQRobertaForSequenceClassification


logging.getLogger().setLevel(logging.WARNING)

num_labels = 3
model_checkpoint = "roberta-base"
batch_size = 32
validation_key = "validation_matched"
dataset = load_dataset("glue", "mnli")
metric = load_metric("glue", "mnli")
nb_step = 1000
training_strategy = IntervalStrategy.STEPS

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=256)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def fuse_qkv(model, quant_per_tensor: bool):
    """Adjust quantization ranges to match an implementation where the QKV projections are implemented with a single GEMM.
    Force the weight and output scale factors to match by taking the max of (Q,K,V).
    """

    def fuse3(qq, qk, qv):
        for mod in [qq, qk, qv]:
            if not hasattr(mod, "_amax"):
                print("          WARNING: NO AMAX BUFFER")
                return
        q = qq._amax.detach().item()
        k = qk._amax.detach().item()
        v = qv._amax.detach().item()

        amax = max(q, k, v)
        qq._amax.fill_(amax)
        qk._amax.fill_(amax)
        qv._amax.fill_(amax)
        print(f"          q={q:5.2f} k={k:5.2f} v={v:5.2f} -> {amax:5.2f}")

    for name, mod in model.named_modules():
        if name.endswith(".attention.self"):
            print(f"FUSE_QKV: {name}")
            fuse3(mod.matmul_q_input_quantizer, mod.matmul_k_input_quantizer, mod.matmul_v_input_quantizer)
            if quant_per_tensor:
                fuse3(mod.query._weight_quantizer, mod.key._weight_quantizer, mod.value._weight_quantizer)


encoded_dataset = dataset.map(preprocess_function, batched=True)

args = TrainingArguments(
    f"{model_checkpoint}-finetuned",
    evaluation_strategy=training_strategy,
    eval_steps=nb_step,
    logging_steps=nb_step,
    save_steps=nb_step,
    save_strategy=training_strategy,
    learning_rate=1e-5,  # 7.5e-6 https://github.com/pytorch/fairseq/issues/2057#issuecomment-643674771
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    num_train_epochs=1,
    fp16=True,
    group_by_length=False,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

model_roberta: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels
)
model_roberta = model_roberta.cuda()

trainer = Trainer(
    model_roberta,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print(trainer.evaluate())
# {'eval_loss': 0.3559744358062744, 'eval_accuracy': 0.8655119714722364, 'eval_runtime': 19.6678, 'eval_samples_per_second': 499.04, 'eval_steps_per_second': 7.83, 'epoch': 0.98}
trainer.train()
trainer.save_model("roberta-model")
del model_roberta
del trainer

input_desc = QuantDescriptor(num_bits=8, calib_method="histogram")
# below we do per-channel quantization for weights, set axis to None to get a per tensor calibration
weight_desc = QuantDescriptor(num_bits=8, axis=(0,))
quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)

# keep it on CPU
model_roberta_q: PreTrainedModel = QDQRobertaForSequenceClassification.from_pretrained("roberta-model")

# Find the TensorQuantizer and enable calibration
for name, module in tqdm(model_roberta_q.named_modules()):
    if isinstance(module, quant_nn.TensorQuantizer):
        if module._calibrator is not None:
            module.disable_quant()
            module.enable_calib()
        else:
            module.disable()

with torch.no_grad():
    for start_index in tqdm(range(0, 128, batch_size)):
        end_index = start_index + batch_size
        data = encoded_dataset["train"][start_index:end_index]
        input_torch = {
            k: torch.tensor(list(v), dtype=torch.long, device="cpu")
            for k, v in data.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        model_roberta_q(**input_torch)


# Finalize calibration
for name, module in model_roberta_q.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
        if module._calibrator is not None:
            if isinstance(module._calibrator, calib.MaxCalibrator):
                module.load_calib_amax()
            else:
                module.load_calib_amax("percentile", percentile=99.99)
            module.enable_quant()
            module.disable_calib()
        else:
            module.enable()

model_roberta_q.cuda()

model_roberta_q.save_pretrained("roberta-trained-quantized")
del model_roberta_q


model_roberta_q: PreTrainedModel = QDQRobertaForSequenceClassification.from_pretrained(
    "roberta-trained-quantized", num_labels=num_labels
)
model_roberta_q = model_roberta_q.cuda()

args.learning_rate /= 10
print(f"LR: {args.learning_rate}")
trainer = Trainer(
    model_roberta_q,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print(trainer.evaluate())
# 4 batches
# {'eval_loss': 0.38076257705688477, 'eval_accuracy': 0.8552215995924605, 'eval_runtime': 46.9577, 'eval_samples_per_second': 209.018, 'eval_steps_per_second': 3.28}
# 100 batches
# {'eval_loss': 0.386756956577301, 'eval_accuracy': 0.8516556291390729, 'eval_runtime': 48.9996, 'eval_samples_per_second': 200.308, 'eval_steps_per_second': 3.143}
trainer.train()
print(trainer.evaluate())
# {'eval_loss': 0.40235549211502075, 'eval_accuracy': 0.8589913397860418, 'eval_runtime': 46.1754, 'eval_samples_per_second': 212.559, 'eval_steps_per_second': 3.335, 'epoch': 1.0}
model_roberta_q.save_pretrained("roberta-in-bert-trained-quantized-retrained")


# fuse_qkv(model_roberta_q, quant_per_tensor=True)
data = encoded_dataset["train"][1:3]
input_torch = {
    k: torch.tensor(list(v), dtype=torch.long, device="cuda")
    for k, v in data.items()
    if k in ["input_ids", "attention_mask", "token_type_ids"]
}

from pytorch_quantization.nn import TensorQuantizer


TensorQuantizer.use_fb_fake_quant = True
convert_to_onnx(model_pytorch=model_roberta_q, output_path="roberta_q.onnx", inputs_pytorch=input_torch)
TensorQuantizer.use_fb_fake_quant = False
# /usr/src/tensorrt/bin/trtexec --onnx=roberta_q.onnx --shapes=input_ids:1x384,attention_mask:1x384 --best --workspace=6000
# no fusing
# Latency: min = 1.85529 ms, max = 4.32666 ms, mean = 1.98449 ms, median = 1.87964 ms, percentile(99%) = 3.19434 ms
# with fusing
# Latency: min = 1.84412 ms, max = 2.22266 ms, mean = 1.87675 ms, median = 1.8717 ms, percentile(99%) = 2.07849 ms
