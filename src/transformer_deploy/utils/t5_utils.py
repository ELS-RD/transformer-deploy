import gc
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
from onnxruntime import InferenceSession, IOBinding
from torch.nn import Linear
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.onnx_utils import merge_autoregressive_model_graphs, save_onnx
from transformer_deploy.backends.ort_utils import (
    add_output_nodes,
    convert_fp16,
    create_model_for_provider,
    get_keep_fp32_nodes,
    inference_onnx_binding,
)
from transformer_deploy.backends.pytorch_utils import convert_to_onnx
from transformer_deploy.backends.trt_utils import TensorRTShape


fp16_default_tolerance = 0.1


class ExportT5(torch.nn.Module):
    def __init__(self, decoder: T5Stack, lm_head: Linear, model_dim):
        super(ExportT5, self).__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.model_dim = model_dim

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, past_key_values: Tuple = None):
        out_dec = self.decoder.forward(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values
        )
        # weight tying -> rescale output before projecting on vocab
        # to comment for T0 for instance
        out_dec["last_hidden_state"] = out_dec["last_hidden_state"] * (self.model_dim ** -0.5)
        out_dec["last_hidden_state"] = self.lm_head(out_dec["last_hidden_state"])
        return out_dec


# https://github.com/NVIDIA/TensorRT/blob/main/demo/HuggingFace/T5/export.py
class ExtT5(torch.nn.Module, GenerationMixin):
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        encoder_path: str,
        decoder_path: str,
    ):
        super(ExtT5, self).__init__()
        self.main_input_name = "input_ids"  # https://github.com/huggingface/transformers/pull/14803
        self.config: PretrainedConfig = config
        self.device: torch.device = device

        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        self.encoder_onnx = create_model_for_provider(encoder_path, provider, log_severity=3)
        self.decoder_onnx = create_model_for_provider(decoder_path, provider, log_severity=3)
        self.use_cache = True
        self.timings = list()

    def encoder_onnx_inference(self, input_ids: torch.Tensor, **_) -> BaseModelOutputWithPastAndCrossAttentions:
        last_hidden_state = inference_onnx_binding(
            model_onnx=self.encoder_onnx,  # noqa: F821
            inputs={"input_ids": input_ids},
            device="cuda",
            binding=self.encoder_onnx.io_binding(),
        )["output"]
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state.type(torch.float16))

    def decoder_onnx_inference(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        enable_cache: torch.Tensor,
        num_layers: int,
        past_key_values: Optional[torch.Tensor],
    ):
        inputs_onnx_dict = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "enable_cache": enable_cache,
        }

        if past_key_values is not None:
            for index, (k_dec, v_dec, k_enc, v_enc) in enumerate(past_key_values):
                inputs_onnx_dict[f"past_key_values.{index}.decoder.key"] = k_dec
                inputs_onnx_dict[f"past_key_values.{index}.decoder.value"] = v_dec
                inputs_onnx_dict[f"past_key_values.{index}.encoder.key"] = k_enc
                inputs_onnx_dict[f"past_key_values.{index}.encoder.value"] = v_enc

        result_dict = inference_onnx_binding(
            model_onnx=self.decoder_onnx,
            inputs=inputs_onnx_dict,
            binding=self.decoder_onnx.io_binding(),  # recycle the binding
            device="cuda",
            clone_tensor=False,  # no memory copy -> best perf and lowest memory footprint!
        )
        past_states = list()
        for index in range(num_layers):
            kv = (
                result_dict[f"present.{index}.decoder.key"],
                result_dict[f"present.{index}.decoder.value"],
                result_dict[f"present.{index}.encoder.key"],
                result_dict[f"present.{index}.encoder.value"],
            )
            past_states.append(kv)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=result_dict["logits"],
            past_key_values=past_states,
        )

    def get_encoder(self):
        return self.encoder_onnx_inference

    def get_decoder(self):
        return self.decoder_onnx_inference

    def set_cache(self, enable: bool) -> None:
        self.use_cache = enable

    # from transformers library (modeling_t5.py)
    def _reorder_cache(self, past, beam_idx):
        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def prepare_inputs_for_generation(self, input_ids, past=None, use_cache=None, **kwargs) -> Dict[str, torch.Tensor]:
        params = {
            "encoder_hidden_states": kwargs["encoder_outputs"]["last_hidden_state"],
        }
        if past is None:  # this is the 1st inferred token
            self.timings = list()
        if not self.use_cache:
            past = None
        if past is None:
            params[self.main_input_name] = input_ids
            params["enable_cache"] = torch.tensor([False], device="cuda", dtype=torch.bool)
        else:
            params[self.main_input_name] = input_ids[:, -1:]
            params["enable_cache"] = torch.tensor([True], device="cuda", dtype=torch.bool)
            params["past_key_values"] = past

        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        enable_cache: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        **_,
    ):
        start_timer = time.monotonic()
        dec_output = self.get_decoder()(
            decoder_input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            enable_cache=enable_cache,
            past_key_values=past_key_values,
            num_layers=self.config.num_layers,
        )
        self.timings.append(time.monotonic() - start_timer)
        return Seq2SeqLMOutput(logits=dec_output.last_hidden_state, past_key_values=dec_output.past_key_values)


def export_t5_decoder_to_onnx(
    model_pytorch: PreTrainedModel,
    model_decoder: ExportT5,
    decoder_cache_model_path: str,
    decoder_no_cache_model_path: str,
    input_ids: torch.Tensor,
    encoder_outputs,
):
    """
    This function is used to export the t5 decoder to onnx format.
    we export 2 versions of the decoder, one without cache support and one with it.
    """
    # decoder output one step before
    out_decoder_pytorch = model_decoder(
        input_ids=input_ids[:, :-1], encoder_hidden_states=encoder_outputs.last_hidden_state
    )

    model_inputs = {
        "input_ids": input_ids[:, -1:].type(torch.int32),
        "encoder_hidden_states": encoder_outputs.last_hidden_state,
        "past_key_values": out_decoder_pytorch.past_key_values,
    }

    input_names = ["input_ids", "encoder_hidden_states"]
    num_layers = model_pytorch.config.num_layers

    for i in range(num_layers):
        input_names.append(f"past_key_values.{i}.decoder.key")
        input_names.append(f"past_key_values.{i}.decoder.value")
        input_names.append(f"past_key_values.{i}.encoder.key")
        input_names.append(f"past_key_values.{i}.encoder.value")

    output_names = ["logits"]

    for i in range(num_layers):
        output_names.append(f"present.{i}.decoder.key")
        output_names.append(f"present.{i}.decoder.value")
        output_names.append(f"present.{i}.encoder.key")
        output_names.append(f"present.{i}.encoder.value")

    dynamic_axis = {
        "input_ids": {0: "batch", 1: "encoder_sequence"},
        "encoder_hidden_states": {0: "batch", 1: "encoder_sequence"},
        "logits": {0: "batch", 1: "decoder_sequence"},
    }

    for i in range(num_layers):
        dynamic_axis[f"past_key_values.{i}.decoder.key"] = {0: "batch", 2: "past_decoder_sequence"}
        dynamic_axis[f"past_key_values.{i}.decoder.value"] = {0: "batch", 2: "past_decoder_sequence"}
        dynamic_axis[f"past_key_values.{i}.encoder.key"] = {0: "batch", 2: "encoder_sequence_length"}
        dynamic_axis[f"past_key_values.{i}.encoder.value"] = {0: "batch", 2: "encoder_sequence_length"}

        dynamic_axis[f"present.{i}.decoder.key"] = {0: "batch", 2: "decoder_sequence"}
        dynamic_axis[f"present.{i}.decoder.value"] = {0: "batch", 2: "decoder_sequence"}
        dynamic_axis[f"present.{i}.encoder.key"] = {0: "batch", 2: "encoder_sequence_length"}
        dynamic_axis[f"present.{i}.encoder.value"] = {0: "batch", 2: "encoder_sequence_length"}

    # Export of the model with cache support
    with torch.no_grad():
        model_pytorch.config.return_dict = True
        model_pytorch.eval()
        torch.onnx.export(
            model_decoder,
            (model_inputs,),
            f=decoder_cache_model_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axis,
            do_constant_folding=True,
            opset_version=13,
        )
    # Export of the model without cache support
    model_inputs_no_cache = {
        "input_ids": input_ids,
        "encoder_hidden_states": encoder_outputs.last_hidden_state,
    }

    with torch.no_grad():
        model_pytorch.config.return_dict = True
        model_pytorch.eval()
        torch.onnx.export(
            model_decoder,
            (model_inputs_no_cache,),
            f=decoder_no_cache_model_path,
            input_names=list(model_inputs_no_cache.keys()),
            output_names=output_names,
            dynamic_axes={k: v for k, v in dynamic_axis.items() if "past_key_values" not in k},
            do_constant_folding=True,
            opset_version=13,
        )

    del model_pytorch, model_decoder, encoder_outputs
    torch.cuda.empty_cache()
    gc.collect()


def decoder_pytorch_inference(decoder_input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, model_decoder, **_):
    with torch.inference_mode():
        return model_decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden_states)


def are_equal(a: torch.Tensor, b: torch.Tensor, atol: float = fp16_default_tolerance) -> None:
    assert np.allclose(a=a.detach().cpu().numpy(), b=b.detach().cpu().numpy(), atol=atol), f"{a}\n\nVS\n\n{b}"


def prepare_folder(path: str) -> Tuple[str, str]:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    [item.unlink() for item in Path(path).glob("*") if item.is_file()]
    return path + "/model.onnx", path + "/model_fp16.onnx"


def get_random_input_encoder() -> Dict[str, torch.Tensor]:
    import random

    max_seq = 128
    seq_len = random.randint(a=1, b=max_seq)
    batch = max_seq // seq_len
    random_input_ids = torch.randint(low=0, high=13200, size=(batch, seq_len), dtype=torch.int32, device="cuda")
    inputs = {"input_ids": random_input_ids}
    return inputs


def get_random_input_no_cache() -> Dict[str, torch.Tensor]:
    inputs = get_random_input_encoder()
    encoder_fp16_onnx = create_model_for_provider(encoder_fp16_model_path, "CUDAExecutionProvider", log_severity=3)
    encoder_fp16_onnx_binding: IOBinding = encoder_fp16_onnx.io_binding()
    encoder_hidden_states = inference_onnx_binding(
        model_onnx=encoder_fp16_onnx,
        binding=encoder_fp16_onnx_binding,
        inputs=inputs,
        device="cuda",
        clone_tensor=False,
    )["output"]
    # it will serve as input of a FP32 model
    inputs["encoder_hidden_states"] = encoder_hidden_states.type(torch.float32)
    # del encoder_fp16_onnx
    return inputs


def get_random_input_cache() -> Dict[str, torch.Tensor]:
    inputs = get_random_input_no_cache()
    inputs["enable_cache"] = torch.tensor([False], device="cuda")
    decoder_if_ort_model = create_model_for_provider(decoder_if_model_path, "CUDAExecutionProvider", log_severity=3)
    decoder_past_states = inference_onnx_binding(
        model_onnx=decoder_if_ort_model,
        inputs=inputs,
        device="cuda",
        clone_tensor=False,
    )
    for k, v in decoder_past_states.items():
        if "present" not in k:
            continue
        new_k = k.replace("present", "past_key_values")
        inputs[new_k] = v
    batch, _ = inputs["input_ids"].shape
    complement = torch.randint(low=0, high=vocab_size, size=(batch, 1), dtype=torch.int32, device="cuda")
    inputs["input_ids"] = torch.concat(tensors=[inputs["input_ids"], complement], dim=1)
    inputs["enable_cache"] = torch.tensor([True], device="cuda")
    return inputs


def decoder_onnx_inference(
    decoder_input_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    enable_cache: torch.Tensor,
    decoder_onnx: InferenceSession,
    num_layers: int,
    past_key_values: Optional[torch.Tensor],
):
    inputs_onnx_dict = {
        "input_ids": decoder_input_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "enable_cache": enable_cache,
    }

    if past_key_values is not None:
        for index, (k_dec, v_dec, k_enc, v_enc) in enumerate(past_key_values):
            inputs_onnx_dict[f"past_key_values.{index}.decoder.key"] = k_dec
            inputs_onnx_dict[f"past_key_values.{index}.decoder.value"] = v_dec
            inputs_onnx_dict[f"past_key_values.{index}.encoder.key"] = k_enc
            inputs_onnx_dict[f"past_key_values.{index}.encoder.value"] = v_enc

    result_dict = inference_onnx_binding(
        model_onnx=decoder_onnx,
        inputs=inputs_onnx_dict,
        binding=decoder_onnx.io_binding(),  # recycle the binding
        device=decoder_input_ids.device.type,
        clone_tensor=False,  # no memory copy -> best perf and lowest memory footprint!
    )
    past_states = list()
    for index in range(num_layers):
        kv = (
            result_dict[f"present.{index}.decoder.key"],
            result_dict[f"present.{index}.decoder.value"],
            result_dict[f"present.{index}.encoder.key"],
            result_dict[f"present.{index}.encoder.value"],
        )
        past_states.append(kv)
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=result_dict["logits"],
        past_key_values=past_states,
    )


def convert_t5_to_onnx(
    tokenizer: PreTrainedTokenizer, model_pytorch: torch.nn.Module, input_ids: torch.Tensor, path_dir: str
):
    global vocab_size
    global encoder_fp16_model_path
    global decoder_if_model_path
    vocab_size = tokenizer.vocab_size
    # prepare sub-folders for t5 onnx models (encoder and decoders)
    encoder_model_path, encoder_fp16_model_path = prepare_folder(path=os.path.join(path_dir, "t5-encoder"))
    decoder_cache_model_path, decoder_cache_fp16_model_path = prepare_folder(
        path=os.path.join(path_dir, "t5-decoder-cache")
    )
    decoder_no_cache_model_path, decoder_no_cache_fp16_model_path = prepare_folder(
        path=os.path.join(path_dir, "t5-decoder-no-cache")
    )
    decoder_if_model_path, decoder_if_fp16_model_path = prepare_folder(path=os.path.join(path_dir, "t5-dec-if-node"))
    # create original outputs to compare it with converted model results:
    encoder_outputs: BaseModelOutputWithPastAndCrossAttentions = model_pytorch.encoder(input_ids=input_ids)
    t5_outputs: Seq2SeqLMOutput = model_pytorch(input_ids=input_ids, decoder_input_ids=input_ids)
    # I. Export to ONNX
    # 1. Export encoder part
    convert_to_onnx(
        model_pytorch=model_pytorch.encoder,
        output_path=encoder_model_path,
        inputs_pytorch={"input_ids": input_ids},
        var_output_seq=True,
        quantization=False,
        output_names=["output"],
    )
    # 2. Export decoder part
    """
    * We first need to wrap it in a Pytorch model to add the final layer so it's output provide scores for
    each vocabulary token and can be directly used by the Hugging Face decoding algorithm
    * Then, we need to manipulate the ONNX graph to add support of Key/Value cache
    """
    model_decoder = ExportT5(
        decoder=model_pytorch.decoder, lm_head=model_pytorch.lm_head, model_dim=model_pytorch.model_dim
    ).eval()
    out_model_export: torch.Tensor = model_decoder(
        input_ids=input_ids, encoder_hidden_states=encoder_outputs.last_hidden_state
    )
    are_equal(a=out_model_export["last_hidden_state"], b=t5_outputs.logits)
    export_t5_decoder_to_onnx(
        model_pytorch, model_decoder, decoder_cache_model_path, decoder_no_cache_model_path, input_ids, encoder_outputs
    )
    # II. Conversion to mixed precision
    # 1. Convert encoder part
    """
    We inject random input sequences and audit the output of each computation graph node; finally,
    we make a list of all nodes that have output values out of the FP16 range /close to zero values
    and perform some cleaning. To finish, we provide the list of nodes to keep in FP32 to the
    conversion function.
    """
    keep_fp32_encoder = get_keep_fp32_nodes(onnx_model_path=encoder_model_path, get_input=get_random_input_encoder)
    assert len(keep_fp32_encoder) > 0
    encoder_model_onnx = convert_fp16(onnx_model=encoder_model_path, nodes_to_exclude=keep_fp32_encoder)
    save_onnx(proto=encoder_model_onnx, model_path=encoder_fp16_model_path, clean=False)

    del encoder_model_onnx
    torch.cuda.empty_cache()
    gc.collect()

    # Compare the output of the ONNX FP16 model with Pytorch one
    encoder_fp16_onnx = create_model_for_provider(encoder_fp16_model_path, "CUDAExecutionProvider", log_severity=3)
    encoder_fp16_onnx_binding = encoder_fp16_onnx.io_binding()
    encoder_onnx_out = inference_onnx_binding(
        model_onnx=encoder_fp16_onnx,
        binding=encoder_fp16_onnx_binding,
        inputs={"input_ids": input_ids},
        device=input_ids.device.type,
    )["output"]
    # are_equal(a=encoder_onnx_out, b=encoder_outputs.last_hidden_state)

    # 2. Convert decoder part
    # Conversion of the decoder module without cache support
    keep_fp32_no_cache = get_keep_fp32_nodes(
        onnx_model_path=decoder_no_cache_model_path, get_input=get_random_input_no_cache
    )
    onnx_model_no_cache_fp16 = convert_fp16(onnx_model=decoder_no_cache_model_path, nodes_to_exclude=keep_fp32_no_cache)
    save_onnx(proto=onnx_model_no_cache_fp16, model_path=decoder_no_cache_fp16_model_path, clean=False)
    del onnx_model_no_cache_fp16
    # Conversion of the decoder module with cache support
    """
    This module requires output from encoder but also from decoder module without cache support
    (as the cache is supposed not to be empty).
    """
    decoder_cache_model: onnx.ModelProto = onnx.load_model(f=decoder_cache_model_path, load_external_data=False)
    decoder_no_cache_model: onnx.ModelProto = onnx.load_model(f=decoder_no_cache_model_path, load_external_data=False)
    assert len(decoder_cache_model.graph.output) == len(decoder_no_cache_model.graph.output)

    decoder_cache_model_fp32_all_nodes = add_output_nodes(model=decoder_cache_model)
    decoder_cache_model_fp32_all_nodes_path = decoder_cache_model_path + "_all_nodes.onnx"
    save_onnx(proto=decoder_cache_model_fp32_all_nodes, model_path=decoder_cache_model_fp32_all_nodes_path, clean=False)
    # reload after shape inference
    decoder_cache_model_fp32_all_nodes = onnx.load_model(
        f=decoder_cache_model_fp32_all_nodes_path, load_external_data=False
    )

    ort_np_type_mapping = {
        onnx.TensorProto.FLOAT: float,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.BOOL: bool,
    }

    # If node requires that the 2 models merged have the exact same number/type of output nodes
    # Above we added many output nodes to the model with cache support...
    # ... we need to add fake output nodes to the other decoder model.
    no_cache_output_nodes = {item.name: item for item in decoder_no_cache_model.graph.output}

    while decoder_no_cache_model.graph.output:
        decoder_no_cache_model.graph.output.pop()

    nb_outputs_to_create = len(decoder_cache_model_fp32_all_nodes.graph.output)
    nodes_to_be_added = list()
    for i in range(nb_outputs_to_create):
        node_name = decoder_cache_model_fp32_all_nodes.graph.output[i].name
        if node_name in no_cache_output_nodes:
            node_to_insert = no_cache_output_nodes[node_name]
            nodes_to_be_added.append(node_to_insert)
        else:
            fake_node_name = f"output_{node_name}"
            fake_node_ort_type = decoder_cache_model_fp32_all_nodes.graph.output[i].type.tensor_type.elem_type
            fake_node_np_type = ort_np_type_mapping[fake_node_ort_type]
            fake_data = np.array([1.0], dtype=fake_node_np_type)
            fake_node = onnx.helper.make_node(
                op_type="Constant",
                inputs=[],
                outputs=[fake_node_name],
                value=onnx.helper.make_tensor(
                    name="const_tensor",
                    data_type=fake_node_ort_type,
                    dims=fake_data.shape,
                    vals=fake_data.flatten(),
                ),
                name=fake_node_name,
            )
            decoder_no_cache_model.graph.node.append(fake_node)
            nodes_to_be_added.append(onnx.ValueInfoProto(name=fake_node_name))

    decoder_no_cache_model.graph.output.extend(nodes_to_be_added)

    decoder_no_cache_model_fp32_all_nodes_path = decoder_no_cache_model_path + "_all_nodes.onnx"
    save_onnx(proto=decoder_no_cache_model, model_path=decoder_no_cache_model_fp32_all_nodes_path, clean=False)

    # now that each model has the same number of output nodes, we can merge them!
    merge_autoregressive_model_graphs(
        model_cache_path=decoder_cache_model_fp32_all_nodes_path,
        model_no_cache_path=decoder_no_cache_model_fp32_all_nodes_path,
        output_path=decoder_if_model_path,
    )
    del decoder_cache_model_fp32_all_nodes, decoder_no_cache_model, decoder_cache_model
    torch.cuda.empty_cache()
    gc.collect()

    """decoder_if_ort_model = create_model_for_provider(decoder_if_model_path, "CUDAExecutionProvider", log_severity=3)
    keep_fp32_cache = search_fp32_nodes(
        original_model=decoder_if_model_path,
        modified_model_session=decoder_if_ort_model,
        get_input=get_random_input_cache,
        early_stop=1,
    )

    # the output node names are those from the decoder module without cache support
    # basically it's the fake node names we added above, we need to remove the output_ prefix to their names
    keep_fp32_cache = [item.replace("output_", "") for item in keep_fp32_cache]

    del decoder_if_ort_model
    torch.cuda.empty_cache()
    gc.collect()"""

    onnx_model_cache_fp16 = convert_fp16(onnx_model=decoder_cache_model_path, nodes_to_exclude=list())
    save_onnx(proto=onnx_model_cache_fp16, model_path=decoder_cache_fp16_model_path, clean=False)

    del onnx_model_cache_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # Merge ONNX computation graph to deduplicate weights
    """
    Finally, we will merge the 2 decoders together. The idea is simple:
        - we prefix the node / edge names of one of them to avoid naming collision
        - we deduplicate the weights (the same weight matrix will have different names in the 2 models)
        - we join the 2 computation graphs through an If node
        - we generate the ONNX file
    The new model will take a new input, enable_cache. When it contains a True value,
    computation graph with cache support is used.
    """
    merge_autoregressive_model_graphs(
        model_cache_path=decoder_cache_fp16_model_path,
        model_no_cache_path=decoder_no_cache_fp16_model_path,
        output_path=decoder_if_fp16_model_path,
    )

    torch.cuda.empty_cache()
    gc.collect()

    # Check ONNX decoder output: Zero copy output
    """
    Below, we check that the new model output is similar to the ones from Pytorch.
    We use our new implementation of inference call.
    The idea is the following:
        - we ask ONNX Runtime to output a pointer to the CUDA array containing the result of the inference;
        - we use Cupy API to wrap the array and provide information regarding tensor shape and type.
         Cupy doesn't own the data;
        - we use Dlpack support to convert the Cupy tensor to Pytorch, another zero copy process.
    This pipeline is unsafe, as the content of the tensor may change or disappear silently: only ONNX Runtime has the
    control of the array containing the data. It will happen at the next inference call. Because we know that during
    the text generation we discard each output before recalling ONNX Runtime, it works well in our case.
    A second benefit of this approach is that we do not have anymore to guess the output shape.
    """
    model_pytorch = model_pytorch.cuda()
    model_decoder = model_decoder.cuda()
    input_ids = input_ids.cuda()
    model_pytorch = model_pytorch.eval()
    model_decoder = model_decoder.eval()
    with torch.inference_mode():
        out_encoder_pytorch: BaseModelOutputWithPastAndCrossAttentions = model_pytorch.encoder(input_ids=input_ids)
        previous_step_pytorch: BaseModelOutputWithPastAndCrossAttentions = model_decoder(
            input_ids=input_ids[:, :-1], encoder_hidden_states=out_encoder_pytorch.last_hidden_state
        )
        out_decoder_pytorch: BaseModelOutputWithPastAndCrossAttentions = model_decoder(
            input_ids=input_ids, encoder_hidden_states=out_encoder_pytorch.last_hidden_state
        )
    model_pytorch = model_pytorch.cpu()
    torch.cuda.empty_cache()
    decoder_onnx = create_model_for_provider(decoder_if_fp16_model_path, "CUDAExecutionProvider", log_severity=3)
    out_decoder_onnx_no_cache = decoder_onnx_inference(
        decoder_input_ids=input_ids,
        encoder_hidden_states=out_encoder_pytorch.last_hidden_state.half(),
        enable_cache=torch.tensor([False], device="cuda", dtype=torch.bool),
        past_key_values=None,
        decoder_onnx=decoder_onnx,
        num_layers=model_pytorch.config.num_layers,
    )
    are_equal(
        a=out_decoder_onnx_no_cache.last_hidden_state[:, -1:, :],
        b=out_decoder_pytorch.last_hidden_state[:, -1:, :],
    )
    # check that past states are identical between ONNX and Pytorch
    assert len(out_decoder_onnx_no_cache.past_key_values) == len(out_decoder_pytorch.past_key_values)
    for (o_dec_k, o_dev_v, o_enc_k, o_enc_v), (p_dec_k, p_dev_v, p_enc_k, p_enc_v) in zip(
        out_decoder_onnx_no_cache.past_key_values, out_decoder_pytorch.past_key_values
    ):
        are_equal(a=o_dec_k, b=p_dec_k)
        are_equal(a=o_dev_v, b=p_dev_v)
        are_equal(a=o_enc_k, b=p_enc_k)
        are_equal(a=o_enc_v, b=p_enc_v)
    # convert ONNX inputs to FP16
    previous_step_pytorch.past_key_values = tuple(
        [tuple([past.half() for past in layer_state]) for layer_state in previous_step_pytorch.past_key_values]
    )
    out_encoder_pytorch.last_hidden_state = out_encoder_pytorch.last_hidden_state.half()

    out_decoder_onnx_cache = decoder_onnx_inference(
        decoder_input_ids=input_ids[:, -1:],
        encoder_hidden_states=out_encoder_pytorch.last_hidden_state,
        enable_cache=torch.tensor([True], device="cuda", dtype=torch.bool),
        past_key_values=previous_step_pytorch.past_key_values,
        decoder_onnx=decoder_onnx,
        num_layers=model_pytorch.config.num_layers,
    )

    are_equal(
        a=out_decoder_onnx_cache.last_hidden_state[:, -1:, :],
        b=out_decoder_pytorch.last_hidden_state[:, -1:, :],
    )

    # check that past states are identical between ONNX and Pytorch
    assert len(out_decoder_onnx_cache.past_key_values) == len(out_decoder_pytorch.past_key_values)
    for (o_dec_k, o_dev_v, o_enc_k, o_enc_v), (p_dec_k, p_dev_v, p_enc_k, p_enc_v) in zip(
        out_decoder_onnx_cache.past_key_values, out_decoder_pytorch.past_key_values
    ):
        are_equal(a=o_dec_k, b=p_dec_k)
        are_equal(a=o_dev_v, b=p_dev_v)
        are_equal(a=o_enc_k, b=p_enc_k)
        are_equal(a=o_enc_v, b=p_enc_v)


def prepare_input_shapes_tensorrt_decoder(input_ids: torch.tensor, num_layers: int) -> List[str]:
    input_ids_shape = input_ids.shape[0]
    input_id_shape = TensorRTShape(
        min_shape=[input_ids_shape, 1],
        optimal_shape=[input_ids_shape, 1],
        max_shape=[input_ids_shape, 256],
        input_name="input_ids",
    )
    encoder_hidden_states_shape = TensorRTShape(
        min_shape=[input_ids_shape, 1, 512],
        optimal_shape=[input_ids_shape, 10, 512],
        max_shape=[input_ids_shape, 200, 512],
        input_name="encoder_hidden_states",
    )

    final_seq_len = TensorRTShape(
        min_shape=[1],
        optimal_shape=[1],
        max_shape=[1],
        input_name="final_seq_len",
    )

    input_shapes = [input_id_shape, encoder_hidden_states_shape, final_seq_len]
    for i in range(num_layers):
        input_shapes.append(
            TensorRTShape(
                min_shape=[input_ids_shape, 8, 0, 64],
                optimal_shape=[input_ids_shape, 8, 100, 64],
                max_shape=[input_ids_shape, 8, 200, 64],
                input_name=f"past_key_values.{i}.decoder.key",
            )
        )
        input_shapes.append(
            TensorRTShape(
                min_shape=[input_ids_shape, 8, 0, 64],
                optimal_shape=[input_ids_shape, 8, 100, 64],
                max_shape=[input_ids_shape, 8, 200, 64],
                input_name=f"past_key_values.{i}.decoder.value",
            )
        )
        input_shapes.append(
            TensorRTShape(
                min_shape=[input_ids_shape, 8, 0, 64],
                optimal_shape=[input_ids_shape, 8, 10, 64],
                max_shape=[input_ids_shape, 8, 200, 64],
                input_name=f"past_key_values.{i}.encoder.key",
            )
        )
        input_shapes.append(
            TensorRTShape(
                min_shape=[input_ids_shape, 8, 0, 64],
                optimal_shape=[input_ids_shape, 8, 10, 64],
                max_shape=[input_ids_shape, 8, 200, 64],
                input_name=f"past_key_values.{i}.encoder.value",
            )
        )

    return input_shapes
