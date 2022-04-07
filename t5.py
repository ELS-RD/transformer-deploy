from collections import OrderedDict
from time import time
from typing import Callable, Dict, Set

import numpy as np
import onnx
import tensorrt as trt
import torch
from onnx import ModelProto
from tensorrt import ICudaEngine
from tensorrt.tensorrt import Logger, Runtime
from torch.nn import Linear
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PretrainedConfig, T5ForConditionalGeneration, TensorType
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding, optimize_onnx
from transformer_deploy.backends.pytorch_utils import convert_to_onnx
from transformer_deploy.backends.trt_utils import (
    TensorRTShape,
    add_output_nodes,
    build_engine,
    get_adjency_dict,
    get_fix_fp16_network_func,
    get_list_fp32_nodes,
    load_engine,
    save_engine,
)


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


model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.eval()
model = model.to("cuda")
input_ids: torch.Tensor = tokenizer("Studies show that", return_tensors=TensorType.PYTORCH).input_ids
input_ids = input_ids.to("cuda")

out_enc: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=input_ids)
convert_to_onnx(
    model_pytorch=model.encoder,
    output_path="test-enc.onnx",
    inputs_pytorch={"input_ids": input_ids},
    var_output_seq=True,
    quantization=False,
)
optimize_onnx(
    onnx_path="test-enc.onnx", onnx_optim_model_path="test-enc-opt.onnx", architecture="bert", use_cuda=True, fp16=True
)

enc_onnx = create_model_for_provider("test-enc-opt.onnx", "CUDAExecutionProvider")
enc_onnx_out = inference_onnx_binding(
    model_onnx=enc_onnx,
    inputs={"input_ids": input_ids},
    device=input_ids.device.type,
    output_shape=tuple(input_ids.shape) + (int(model.encoder.config.d_model),),
)["output"]
assert np.allclose(enc_onnx_out.detach().cpu().numpy(), out_enc.last_hidden_state.detach().cpu().numpy(), atol=1e-2)

out_full: Seq2SeqLMOutput = model(input_ids=input_ids, decoder_input_ids=input_ids)
model_to_export = ExportT5(decoder=model.decoder, lm_head=model.lm_head).eval()
out_model_export: torch.Tensor = model_to_export(input_ids=input_ids, encoder_hidden_states=out_enc.last_hidden_state)
assert np.allclose(out_model_export.detach().cpu().numpy(), out_full.logits.detach().cpu().numpy(), atol=1e-5)

inputs_onnx = OrderedDict({"input_ids": input_ids, "encoder_hidden_states": out_enc.last_hidden_state})

convert_to_onnx(
    model_pytorch=model_to_export,
    output_path="test-dec.onnx",
    inputs_pytorch=inputs_onnx,
    var_output_seq=False,
    quantization=False,
    fix_output_dim_size=False,  # specific to decoder part
)
optimize_onnx(
    onnx_path="test-dec.onnx",
    onnx_optim_model_path="test-dec-opt.onnx",
    architecture="bert",
    use_cuda=True,
    fp16=True,
    num_attention_heads=model.config.num_heads,
    hidden_size=model.config.d_model,
)

# import onnx
# onnx_dec_model = onnx.load("test-dec-opt.onnx")
# onnx_dec_model.graph.output[0].type.tensor_type.shape.dim[1].dim_param = "seq1"
# onnx.save(onnx_dec_model, "test-dec-opt.onnx")

dec_onnx = create_model_for_provider("test-dec-opt.onnx", "CUDAExecutionProvider")


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
        device=input_ids.device.type,
        output_shape=tuple(input_ids.shape) + (model.config.vocab_size,),
    )
    return result_dict["output"]


def decoder_onnx_standard_inference(input_ids: torch.Tensor, last_hidden_state: torch.Tensor):
    result_list = dec_onnx.run(
        None, {"input_ids": input_ids.type(torch.int32).numpy(), "encoder_hidden_states": last_hidden_state.numpy()}
    )
    return torch.from_numpy(result_list[0])


dec_onnx_out = decoder_onnx_inference(input_ids=input_ids, last_hidden_state=out_enc.last_hidden_state)
assert np.allclose(dec_onnx_out.detach().cpu().numpy(), out_full.logits.detach().cpu().numpy(), atol=1e-1)


def encoder_onnx_inference(input_ids: torch.Tensor, **_) -> BaseModelOutputWithPastAndCrossAttentions:
    result = inference_onnx_binding(
        model_onnx=enc_onnx,
        inputs={"input_ids": input_ids},
        output_shape=tuple(input_ids.shape) + (int(model.encoder.config.d_model),),
        device=input_ids.device.type,
    )
    return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=result["output"])


def encoder_pytorch_inference(input_ids, **_) -> BaseModelOutputWithPastAndCrossAttentions:
    return model.encoder(input_ids=input_ids)


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
            "encoder_hidden_states": kwargs["encoder_outputs"]["last_hidden_state"],
        }

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, **_):
        dec_output = self.get_decoder()(input_ids=input_ids, last_hidden_state=encoder_hidden_states)
        return Seq2SeqLMOutput(logits=dec_output)


model_gen = (
    ExtT5(
        config=model.config,
        device=model.device,
        encoder_func=encoder_onnx_inference,  # encoder_pytorch_inference
        decoder_func=decoder_onnx_inference,  # decoder_pytorch_inference
    )
    .cuda()
    .eval()
)

# model = model.eval()
with torch.inference_mode():
    out_enc: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=input_ids)
    a = model_gen(input_ids=input_ids, encoder_hidden_states=out_enc.last_hidden_state).logits
    b = model(input_ids=input_ids, decoder_input_ids=input_ids).logits
    assert np.allclose(a.detach().cpu().numpy(), b.detach().cpu().numpy(), atol=1e-1)

    print(
        tokenizer.decode(
            model_gen.generate(inputs=input_ids, max_length=20, num_beams=7, no_repeat_ngram_size=2)[0],
            skip_special_tokens=False,
        )
    )
    print(
        tokenizer.decode(
            model.generate(inputs=input_ids, max_length=20, num_beams=7, no_repeat_ngram_size=2)[0],
            skip_special_tokens=False,
        )
    )

start = time()
for _ in range(3):
    model_gen.generate(inputs=input_ids, max_length=500, num_beams=5, no_repeat_ngram_size=2, min_length=500)
print(time() - start)

model.config.use_cache = True
with torch.inference_mode():
    start = time()
    for _ in range(3):
        model.generate(inputs=input_ids, max_length=500, num_beams=5, no_repeat_ngram_size=2, min_length=500)
    print(time() - start)

model = model.cpu()
del enc_onnx
del dec_onnx

trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
runtime: Runtime = trt.Runtime(trt_logger)
trt_model_name = "trt-t5-dec.plan"

# create only of does not exist because it's slow to run...

# 768 for base model, 512 for small, make it dependent from the Pytorch model configuration
input_id_shape = TensorRTShape(min_shape=[5, 1], optimal_shape=[5, 500], max_shape=[5, 500], input_name="input_ids")
encoder_hidden_states_shape = TensorRTShape(
    min_shape=[5, 1, 512], optimal_shape=[5, 500 // 2, 512], max_shape=[5, 500, 512], input_name="encoder_hidden_states"
)


model = model.cuda()
model_onnx: ModelProto = onnx.load("test-dec.onnx")
model_onnx_all_nodes = add_output_nodes(model=model_onnx)
onnx_graph: Dict[str, Set[str]] = get_adjency_dict(model=model_onnx)
ort_model_all_nodes = create_model_for_provider(model_onnx_all_nodes.SerializeToString(), "CUDAExecutionProvider")


# use info from tokenizer size and max shape provided through the command line
def get_random_input():
    input = torch.randint(high=tokenizer.vocab_size, size=(5, 500), dtype=torch.int32, device="cuda")
    hidden_state = model.encoder(input_ids=input).last_hidden_state.detach().cpu().numpy()
    return {"input_ids": input.detach().cpu().numpy(), "encoder_hidden_states": hidden_state}


keep_fp32 = get_list_fp32_nodes(
    onnx_graph=onnx_graph, model=ort_model_all_nodes, get_input=get_random_input, nb_try=200
)
model = model.cpu()

engine: ICudaEngine = build_engine(
    runtime=runtime,
    onnx_file_path="test-dec.onnx",
    logger=trt_logger,
    workspace_size=20000 * 1024**2,
    fp16=True,
    int8=False,
    input_shapes=[input_id_shape, encoder_hidden_states_shape],
    fp16_fix=get_fix_fp16_network_func(keep_fp32=keep_fp32),
)
save_engine(engine, trt_model_name)

tensorrt_model = load_engine(runtime=runtime, engine_file_path=trt_model_name)
a = tensorrt_model(
    {
        "input_ids": input_ids.type(torch.int32).repeat((5, 1)),
        "encoder_hidden_states": out_enc.last_hidden_state.repeat((5, 1, 1)),
    }
)
print(a[0][0])

benchmark_input = torch.ones((5, 500), dtype=torch.int32, device="cuda")
benchmark_enc_output = out_enc.last_hidden_state.repeat((5, 1, 1))
for _ in range(10):
    tensorrt_model(
        {
            "input_ids": benchmark_input,
            "encoder_hidden_states": benchmark_enc_output,
        }
    )
start = time()
for _ in range(100):
    tensorrt_model(
        {
            "input_ids": benchmark_input,
            "encoder_hidden_states": benchmark_enc_output,
        }
    )
print(time() - start)

dec_onnx = create_model_for_provider("test-dec-opt.onnx", "CUDAExecutionProvider")
dec_onnx_out = decoder_onnx_inference(input_ids=input_ids, last_hidden_state=out_enc.last_hidden_state)


for _ in range(10):
    decoder_onnx_inference(input_ids=benchmark_input, last_hidden_state=benchmark_enc_output)
start = time()
for _ in range(100):
    decoder_onnx_inference(input_ids=benchmark_input, last_hidden_state=benchmark_enc_output)
print(time() - start)

model.cuda()
for _ in range(10):
    model.decoder(input_ids=benchmark_input, encoder_hidden_states=benchmark_enc_output)
start = time()
for _ in range(100):
    model.decoder(input_ids=benchmark_input, encoder_hidden_states=benchmark_enc_output)
print(time() - start)

# TensorRT, ONNX Runtime, Pytorch

# sequence 500
# 0.8640644550323486
# 0.6695075035095215
# 1.1308434009552002

# sequence 250
# 0.9177014827728271
# 0.6861860752105713
# 1.1923034191131592
