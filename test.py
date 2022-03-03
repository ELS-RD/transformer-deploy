from collections import OrderedDict

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
        # assert input_ids.dtype == torch.int32 or input_ids.dtype == torch.int64
        # assert encoder_hidden_states.dtype == torch.float32, f"{encoder_hidden_states.dtype}"
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
enc_onnx_out = inference_onnx_binding(model_onnx=enc_onnx, inputs={"input_ids": input_ids}, device="cpu")
assert np.allclose(enc_onnx_out["output"].detach().numpy(), out_enc.last_hidden_state.detach().numpy(), atol=1e-5)


out_full: Seq2SeqLMOutput = model(input_ids=input_ids, decoder_input_ids=input_ids)
model_to_export = ExportT5(decoder=model.decoder, lm_head=model.lm_head)
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

dec_onnx = create_model_for_provider("test-dec.onnx", "CPUExecutionProvider")
dec_onnx_out = inference_onnx_binding(model_onnx=dec_onnx, inputs=inputs_onnx, device="cpu")
assert np.allclose(dec_onnx_out["output"].detach().numpy(), out_full.logits.detach().numpy(), atol=1e-5)


# https://github.com/NVIDIA/TensorRT/blob/main/demo/HuggingFace/T5/export.py
class ExtT5(torch.nn.Module, GenerationMixin):
    def __init__(
        self, config: PretrainedConfig, device: torch.device, encoder: T5Stack, decoder: T5Stack, lm_head: Linear
    ):
        super(ExtT5, self).__init__()
        self.config: PretrainedConfig = config
        self.device: torch.device = device

        self.encoder = encoder
        self.decoder = decoder
        self.lm_head = lm_head
        self.main_input_name = "input_ids"  # https://github.com/huggingface/transformers/pull/14803

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            self.main_input_name: input_ids,
            "encoder_hidden_states": kwargs["encoder_hidden_states"],
        }

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, **_):
        out_dec = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)["last_hidden_state"]
        # Rescale output before projecting on vocab
        out_dec = out_dec * (model.model_dim**-0.5)
        out_lm = self.lm_head(out_dec)
        return Seq2SeqLMOutput(logits=out_lm)


model_gen = ExtT5(
    config=model.config, device=model.device, encoder=model.encoder, decoder=model.decoder, lm_head=model.lm_head
)
model_gen = model_gen.eval()
model = model.eval()

with torch.inference_mode():
    a = model_gen(input_ids=input_ids, encoder_hidden_states=out_enc.last_hidden_state).logits
    b = model(input_ids=input_ids, decoder_input_ids=input_ids).logits
    assert np.alltrue(a.detach().cpu().numpy() == b.detach().cpu().numpy())

    print(
        tokenizer.decode(
            model_gen.generate(inputs=input_ids, encoder_hidden_states=out_enc.last_hidden_state, max_length=20)[0],
            skip_special_tokens=False,
        )
    )
    print(tokenizer.decode(model.generate(inputs=input_ids, max_length=20)[0], skip_special_tokens=False))
