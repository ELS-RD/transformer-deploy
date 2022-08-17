import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import onnx
import torch
from onnxruntime import InferenceSession, IOBinding
from torch.nn import Linear
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizer, TensorType
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack

from transformer_deploy.backends.onnx_utils import merge_autoregressive_model_graphs, save_onnx
from transformer_deploy.backends.ort_utils import (
    add_output_nodes,
    convert_fp16,
    create_model_for_provider,
    get_keep_fp32_nodes,
    inference_onnx_binding,
    search_fp32_nodes,
)
from transformer_deploy.backends.pytorch_utils import convert_to_onnx


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
    out_dec_pytorch = model_decoder(
        input_ids=input_ids[:, :-1], encoder_hidden_states=encoder_outputs.last_hidden_state
    )

    model_inputs = {
        "input_ids": input_ids[:, -1:].type(torch.int32),
        "encoder_hidden_states": encoder_outputs.last_hidden_state,
        "past_key_values": out_dec_pytorch.past_key_values,
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


def decoder_pytorch_inference(decoder_input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, model_decoder, **_):
    with torch.inference_mode():
        return model_decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden_states)


def decoder_onnx_inference(
    model_pytorch: PreTrainedModel,
    decoder_input_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    enable_cache: torch.Tensor,
    decoder_onnx: InferenceSession,
    decoder_onnx_binding: IOBinding,
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
        binding=decoder_onnx_binding,  # recycle the binding
        device=decoder_input_ids.device.type,
        clone_tensor=False,  # no memory copy -> best perf and lowest memory footprint!
    )
    past_states = list()
    for index in range(model_pytorch.config.num_layers):
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
    _, encoder_fp16_model_path = prepare_folder(path="./test-enc")
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
    del encoder_fp16_onnx
    return inputs


def get_random_input_cache(
    tokenizer: PreTrainedTokenizer, decoder_if_ort_model: InferenceSession
) -> Dict[str, torch.Tensor]:
    inputs = get_random_input_no_cache()
    inputs["enable_cache"] = torch.tensor([False], device="cuda")
    dec_past_states = inference_onnx_binding(
        model_onnx=decoder_if_ort_model,
        inputs=inputs,
        device="cuda",
        clone_tensor=False,
    )
    for k, v in dec_past_states.items():
        if "present" not in k:
            continue
        new_k = k.replace("present", "past_key_values")
        inputs[new_k] = v
    batch, _ = inputs["input_ids"].shape
    complement = torch.randint(low=0, high=tokenizer.vocab_size, size=(batch, 1), dtype=torch.int32, device="cuda")
    inputs["input_ids"] = torch.concat(tensors=[inputs["input_ids"], complement], dim=1)
    inputs["enable_cache"] = torch.tensor([True], device="cuda")
    return inputs


def convert_t5(tokenizer: PreTrainedTokenizer, model_name: str, auth_token: str):
    """global vocab_size
    vocab_size = tokenizer.vocab_size"""
    # prepare sub-folders for t5 onnx models (encoder and decoders)
    encoder_model_path, encoder_fp16_model_path = prepare_folder(path="./test-enc")
    decoder_cache_model_path, decoder_cache_fp16_model_path = prepare_folder(path="./test-dec-cache")
    decoder_no_cache_model_path, decoder_no_cache_fp16_model_path = prepare_folder(path="./test-dec-no-cache")
    decoder_if_model_path, decoder_if_fp16_model_path = prepare_folder(path="./test-dec-if")
    input_ids: torch.Tensor = tokenizer(
        'translate English to French: Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new "Colossal Clean Crawled Corpus", we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code.',
        return_tensors=TensorType.PYTORCH,
    ).input_ids
    input_ids = input_ids.type(torch.int32)
    model_pytorch = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=auth_token)
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
    """
    encoder_fp16_onnx = create_model_for_provider(encoder_fp16_model_path, "CUDAExecutionProvider", log_severity=3)
    encoder_onnx_out = inference_onnx_binding(
        model_onnx=encoder_fp16_onnx,
        binding=encoder_fp16_onnx_binding,
        inputs={"input_ids": input_ids},
        device=input_ids.device.type,
    )["output"]
    are_equal(a=encoder_onnx_out, b=encoder_outputs.last_hidden_state)
    """

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
    dec_cache_model: onnx.ModelProto = onnx.load_model(f=decoder_cache_model_path, load_external_data=False)
    dec_no_cache_model: onnx.ModelProto = onnx.load_model(f=decoder_no_cache_model_path, load_external_data=False)
    assert len(dec_cache_model.graph.output) == len(dec_no_cache_model.graph.output)

    dec_cache_model_fp32_all_nodes = add_output_nodes(model=dec_cache_model)
    dec_cache_model_fp32_all_nodes_path = decoder_cache_model_path + "_all_nodes.onnx"
    save_onnx(proto=dec_cache_model_fp32_all_nodes, model_path=dec_cache_model_fp32_all_nodes_path, clean=False)
    # reload after shape inference
    dec_cache_model_fp32_all_nodes = onnx.load_model(f=dec_cache_model_fp32_all_nodes_path, load_external_data=False)

    ort_np_type_mapping = {
        onnx.TensorProto.FLOAT: float,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.BOOL: bool,
    }

    # If node requires that the 2 models merged have the exact same number/type of output nodes
    # Above we added many output nodes to the model with cache support...
    # ... we need to add fake output nodes to the other decoder model.
    no_cache_output_nodes = {item.name: item for item in dec_no_cache_model.graph.output}

    while dec_no_cache_model.graph.output:
        dec_no_cache_model.graph.output.pop()

    nb_outputs_to_create = len(dec_cache_model_fp32_all_nodes.graph.output)
    nodes_to_be_added = list()
    for i in range(nb_outputs_to_create):
        node_name = dec_cache_model_fp32_all_nodes.graph.output[i].name
        if node_name in no_cache_output_nodes:
            node_to_insert = no_cache_output_nodes[node_name]
            nodes_to_be_added.append(node_to_insert)
        else:
            fake_node_name = f"output_{node_name}"
            fake_node_ort_type = dec_cache_model_fp32_all_nodes.graph.output[i].type.tensor_type.elem_type
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
            dec_no_cache_model.graph.node.append(fake_node)
            nodes_to_be_added.append(onnx.ValueInfoProto(name=fake_node_name))

    dec_no_cache_model.graph.output.extend(nodes_to_be_added)

    dec_no_cache_model_fp32_all_nodes_path = decoder_no_cache_model_path + "_all_nodes.onnx"
    save_onnx(proto=dec_no_cache_model, model_path=dec_no_cache_model_fp32_all_nodes_path, clean=False)

    # now that each model has the same number of output nodes, we can merge them!
    merge_autoregressive_model_graphs(
        model_cache_path=dec_cache_model_fp32_all_nodes_path,
        model_no_cache_path=dec_no_cache_model_fp32_all_nodes_path,
        output_path=decoder_if_model_path,
    )
    del dec_cache_model_fp32_all_nodes, dec_no_cache_model, dec_cache_model

    torch.cuda.empty_cache()
    gc.collect()
    decoder_if_ort_model = create_model_for_provider(decoder_if_model_path, "CUDAExecutionProvider", log_severity=3)
    keep_fp32_cache = search_fp32_nodes(
        original_model=decoder_if_model_path,
        modified_model_session=decoder_if_ort_model,
        get_input=get_random_input_cache,
        early_stop=100,
    )

    # the output node names are those from the decoder module without cache support
    # basically it's the fake node names we added above, we need to remove the output_ prefix to their names
    keep_fp32_cache = [item.replace("output_", "") for item in keep_fp32_cache]

    del decoder_if_ort_model
    torch.cuda.empty_cache()
    gc.collect()

    onnx_model_cache_fp16 = convert_fp16(onnx_model=decoder_cache_model_path, nodes_to_exclude=keep_fp32_cache)
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
        out_enc_pytorch: BaseModelOutputWithPastAndCrossAttentions = model_pytorch.encoder(input_ids=input_ids)
        previous_step_pytorch: BaseModelOutputWithPastAndCrossAttentions = model_decoder(
            input_ids=input_ids[:, :-1], encoder_hidden_states=out_enc_pytorch.last_hidden_state
        )
        out_dec_pytorch: BaseModelOutputWithPastAndCrossAttentions = model_decoder(
            input_ids=input_ids, encoder_hidden_states=out_enc_pytorch.last_hidden_state
        )
    model_pytorch = model_pytorch.cpu()
    model_decoder = model_decoder.cpu()
    torch.cuda.empty_cache()
    encoder_fp16_onnx = create_model_for_provider(encoder_fp16_model_path, "CUDAExecutionProvider", log_severity=3)
    encoder_fp16_onnx_binding: IOBinding = encoder_fp16_onnx.io_binding()
    decoder_onnx = create_model_for_provider(decoder_if_fp16_model_path, "CUDAExecutionProvider", log_severity=3)
    decoder_onnx_binding: IOBinding = decoder_onnx.io_binding()
    out_dec_onnx_no_cache = decoder_onnx_inference(
        model_pytorch=model_pytorch,
        decoder_input_ids=input_ids,
        encoder_hidden_states=out_enc_pytorch.last_hidden_state.half(),
        enable_cache=torch.tensor([False], device="cuda", dtype=torch.bool),
        past_key_values=None,
        decoder_onnx=model_decoder,
        decoder_onnx_binding=decoder_onnx_binding,
    )
    are_equal(
        a=out_dec_onnx_no_cache.last_hidden_state[:, -1:, :],
        b=out_dec_pytorch.last_hidden_state[:, -1:, :],
    )
    # check that past states are identical between ONNX and Pytorch
    assert len(out_dec_onnx_no_cache.past_key_values) == len(out_dec_pytorch.past_key_values)
    for (o_dec_k, o_dev_v, o_enc_k, o_enc_v), (p_dec_k, p_dev_v, p_enc_k, p_enc_v) in zip(
        out_dec_onnx_no_cache.past_key_values, out_dec_pytorch.past_key_values
    ):
        are_equal(a=o_dec_k, b=p_dec_k)
        are_equal(a=o_dev_v, b=p_dev_v)
        are_equal(a=o_enc_k, b=p_enc_k)
        are_equal(a=o_enc_v, b=p_enc_v)
    # convert ONNX inputs to FP16
    previous_step_pytorch.past_key_values = tuple(
        [tuple([past.half() for past in layer_state]) for layer_state in previous_step_pytorch.past_key_values]
    )
    out_enc_pytorch.last_hidden_state = out_enc_pytorch.last_hidden_state.half()

    out_dec_onnx_cache = decoder_onnx_inference(
        decoder_input_ids=input_ids[:, -1:],
        encoder_hidden_states=out_enc_pytorch.last_hidden_state,
        enable_cache=torch.tensor([True], device="cuda", dtype=torch.bool),
        past_key_values=previous_step_pytorch.past_key_values,
    )

    are_equal(
        a=out_dec_onnx_cache.last_hidden_state[:, -1:, :],
        b=out_dec_pytorch.last_hidden_state[:, -1:, :],
    )

    # check that past states are identical between ONNX and Pytorch
    assert len(out_dec_onnx_cache.past_key_values) == len(out_dec_pytorch.past_key_values)
    for (o_dec_k, o_dev_v, o_enc_k, o_enc_v), (p_dec_k, p_dev_v, p_enc_k, p_enc_v) in zip(
        out_dec_onnx_cache.past_key_values, out_dec_pytorch.past_key_values
    ):
        are_equal(a=o_dec_k, b=p_dec_k)
        are_equal(a=o_dev_v, b=p_dev_v)
        are_equal(a=o_enc_k, b=p_enc_k)
        are_equal(a=o_enc_v, b=p_enc_v)
