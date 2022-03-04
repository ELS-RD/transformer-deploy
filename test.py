from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
from torch.nn import Linear
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PretrainedConfig, T5ForConditionalGeneration, TensorType
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding
from transformer_deploy.backends.pytorch_utils import convert_to_onnx


class ExportT5(torch.nn.Module):
    def __init__(self, decoder: T5Stack, lm_head: Linear):
        super(ExportT5, self).__init__()
        self.decoder = decoder
        self.lm_head = lm_head

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor):
        out_dec = self.decoder.forward(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
        # Rescale output before projecting on vocab
        out_dec = out_dec["last_hidden_state"] * (model.model_dim**-0.5)
        out_lm = self.lm_head(out_dec)
        return out_lm


model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.eval()

input_ids = tokenizer("Studies show that", return_tensors=TensorType.PYTORCH).input_ids

out_enc: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=input_ids)
convert_to_onnx(
    model_pytorch=model.encoder,
    output_path="test-enc.onnx",
    inputs_pytorch={"input_ids": input_ids},
    var_output_seq=True,
    quantization=False,
)

enc_onnx = create_model_for_provider("test-enc.onnx", "CPUExecutionProvider")
enc_onnx_out = inference_onnx_binding(
    model_onnx=enc_onnx, inputs={"input_ids": input_ids}, device=str(input_ids.device)
)
assert np.allclose(enc_onnx_out["output"].detach().numpy(), out_enc.last_hidden_state.detach().numpy(), atol=1e-5)

out_full: Seq2SeqLMOutput = model(input_ids=input_ids, decoder_input_ids=input_ids)
model_to_export = ExportT5(decoder=model.decoder, lm_head=model.lm_head).eval()
out_model_export: torch.Tensor = model_to_export(input_ids=input_ids, encoder_hidden_states=out_enc.last_hidden_state)
assert np.allclose(out_model_export.detach().numpy(), out_full.logits.detach().numpy(), atol=1e-5)

inputs_onnx = OrderedDict({"input_ids": input_ids, "encoder_hidden_states": out_enc.last_hidden_state})

convert_to_onnx(
    model_pytorch=model_to_export,
    output_path="test-dec.onnx",
    inputs_pytorch=inputs_onnx,
    var_output_seq=True,
    quantization=False,
)


def decoder_pytorch_inference(input_ids: torch.Tensor, last_hidden_state: torch.Tensor):
    out_dec = model.decoder(input_ids=input_ids, encoder_hidden_states=last_hidden_state)["last_hidden_state"]
    # Rescale output before projecting on vocab
    out_dec = out_dec * (model.model_dim**-0.5)
    out_lm = model.lm_head(out_dec)
    return out_lm


def decoder_onnx_inference(input_ids: torch.Tensor, last_hidden_state: torch.Tensor):
    result_dict = inference_onnx_binding(
        model_onnx=dec_onnx,
        inputs={"input_ids": input_ids, "encoder_hidden_states": last_hidden_state},
        device=str(input_ids.device),
        output_shape=tuple(input_ids.shape) + (32128,),
    )
    return result_dict["output"]


dec_onnx = create_model_for_provider("test-dec.onnx", "CPUExecutionProvider")
dec_onnx_out = decoder_onnx_inference(input_ids=input_ids, last_hidden_state=out_enc.last_hidden_state)
assert np.allclose(dec_onnx_out.detach().numpy(), out_full.logits.detach().numpy(), atol=1e-5)


def encoder_onnx_inference(input_ids: torch.Tensor, **_) -> torch.Tensor:
    result = inference_onnx_binding(model_onnx=enc_onnx, inputs={"input_ids": input_ids}, device=str(input_ids.device))
    return result["output"]


def encoder_pytorch_inference(input_ids, **_) -> torch.Tensor:
    return model.encoder(input_ids=input_ids).last_hidden_state


# https://github.com/NVIDIA/TensorRT/blob/main/demo/HuggingFace/T5/export.py
class ExtT5(torch.nn.Module, GenerationMixin):
    def __init__(self, config: PretrainedConfig, device: torch.device, encoder_func: Callable, decoder_func: Callable):
        super(ExtT5, self).__init__()
        self.main_input_name = "input_ids"  # https://github.com/huggingface/transformers/pull/14803
        self.config: PretrainedConfig = config
        self.device: torch.device = device

        self.encoder_func = encoder_func
        self.decoder_func = decoder_func

    def get_encoder(self):
        return self.encoder_func

    def get_decoder(self):
        return self.decoder_func

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            self.main_input_name: input_ids,
            "encoder_hidden_states": kwargs["encoder_outputs"],
        }

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, **_):
        dec_output = self.get_decoder()(input_ids=input_ids, last_hidden_state=encoder_hidden_states)
        return Seq2SeqLMOutput(logits=dec_output)


model_gen = ExtT5(
    config=model.config,
    device=model.device,
    encoder_func=encoder_onnx_inference,  # encoder_pytorch_inference
    decoder_func=decoder_onnx_inference,  # decoder_pytorch_inference
).eval()

# model = model.eval()
with torch.inference_mode():
    out_enc: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=input_ids)
    a = model_gen(input_ids=input_ids, encoder_hidden_states=out_enc.last_hidden_state).logits
    b = model(input_ids=input_ids, decoder_input_ids=input_ids).logits
    assert np.allclose(a.detach().cpu().numpy(), b.detach().cpu().numpy(), atol=1e-5)

    print(tokenizer.decode(model_gen.generate(inputs=input_ids, max_length=20)[0], skip_special_tokens=False))
    print(tokenizer.decode(model.generate(inputs=input_ids, max_length=20)[0], skip_special_tokens=False))
