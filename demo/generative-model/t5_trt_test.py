import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, TensorType
from trt_t5_utils import T5TRT


model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids: torch.Tensor = tokenizer(
    ["translate English to French: This model is now very fast!"],
    return_tensors=TensorType.PYTORCH,
    padding=True,
).input_ids
input_ids = input_ids.to("cuda")
model: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.config.use_cache = True

model = model.to("cuda")
t5_trt_encoder_plan = f"{model_name}-trt-enc.plan"
t5_trt_decoder_plan = f"{model_name}-trt-dec.plan"

t5_trt_wrapper = T5TRT(
    config=model.config,
    encoder_engine_path=t5_trt_encoder_plan,
    decoder_engine_path=t5_trt_decoder_plan,
    tokenizer=tokenizer,
    device=model.device,
    use_cache=True,
)
outputs = t5_trt_wrapper.generate(inputs=input_ids, max_length=8, num_beams=4, no_repeat_ngram_size=2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
