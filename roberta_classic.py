from datasets import load_dataset, load_metric
from tqdm import tqdm
from transformer_deploy.QDQModels.QDQRoberta import QDQRobertaForSequenceClassification

from transformers import AutoTokenizer
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
import numpy as np

import torch
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    IntervalStrategy,
)
from pytorch_quantization import calib

num_labels = 3
model_checkpoint = "roberta-base"
batch_size = 32
validation_key = "validation_matched"
dataset = load_dataset("glue", "mnli")
metric = load_metric('glue', "mnli")
nb_step = 1000
training_strategy = IntervalStrategy.STEPS

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"],
                     truncation=True,
                     padding="max_length",
                     max_length=256)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


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

model_roberta: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model_roberta = model_roberta.cuda()

trainer = Trainer(
    model_roberta,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
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

model_roberta_q: PreTrainedModel = QDQRobertaForSequenceClassification.from_pretrained("roberta-model")
model_roberta_q = model_roberta_q.cuda()
# Find the TensorQuantizer and enable calibration
for name, module in tqdm(model_roberta_q.named_modules()):
    if isinstance(module, quant_nn.TensorQuantizer):
        if module._calibrator is not None:
            module.disable_quant()
            module.enable_calib()
        else:
            module.disable()

with torch.no_grad():
    for start_index in tqdm(range(0, 4*batch_size, batch_size)):
        end_index = start_index + batch_size
        data = encoded_dataset["train"][start_index:end_index]
        input_torch = {k: torch.tensor(list(v), dtype=torch.long, device="cuda")
                       for k, v in data.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
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

model_roberta_q: PreTrainedModel = QDQRobertaForSequenceClassification.from_pretrained("roberta-trained-quantized", num_labels=num_labels)
model_roberta_q = model_roberta_q.cuda()

args.learning_rate /= 10
print(f"LR: {args.learning_rate}")
trainer = Trainer(
    model_roberta_q,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print(trainer.evaluate())
# {'eval_loss': 0.38076257705688477, 'eval_accuracy': 0.8552215995924605, 'eval_runtime': 46.9577, 'eval_samples_per_second': 209.018, 'eval_steps_per_second': 3.28}
trainer.train()
print(trainer.evaluate())
model_roberta_q.save_pretrained("roberta-in-bert-trained-quantized-retrained")
