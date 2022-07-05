#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import tempfile
from typing import List, Optional

import numpy
import numpy as np
import onnx
import pytest
import torch
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, helper
from onnxruntime import InferenceSession, OrtValue
from pytest_benchmark.fixture import BenchmarkFixture

from transformer_deploy.backends.onnx_utils import (
    convert_bf16_to_fp32,
    convert_fp32_to_bf16,
    merge_autoregressive_model_graphs,
    save_onnx,
)
from transformer_deploy.backends.ort_utils import (
    get_io_to_node_mapping,
    inference_onnx_binding,
    ort_conversion_table,
    to_pytorch,
)


def get_simple_onnx(input_type: int) -> ModelProto:
    """
    Generate a very simple ONNX model
    :return: Onnx model
    """
    # I/O
    input_x: ValueInfoProto = helper.make_tensor_value_info("X", input_type, shape=["axis1", "axis2"])
    out: ValueInfoProto = helper.make_tensor_value_info("OUT", input_type, shape=["axis1", "axis2"])

    # Create a node (NodeProto)
    add_1_node: NodeProto = helper.make_node(
        op_type="Add",
        inputs=["X", "Tensor"],
        outputs=["add1"],
        name="first_add",
    )

    add_2_node: NodeProto = helper.make_node(
        op_type="Add",
        inputs=["add1", "Tensor"],
        outputs=["OUT"],
        name="second_add",
    )

    init_tensor = helper.make_tensor("Tensor", dims=(1, 1), vals=[1], data_type=input_type)

    graph_add: GraphProto = helper.make_graph(
        nodes=[add_1_node, add_2_node],
        name="add-model",
        inputs=[input_x],
        outputs=[out],
        initializer=[init_tensor],
    )

    model_def: ModelProto = helper.make_model(
        graph_add, producer_name="onnx-example", opset_imports=[helper.make_opsetid(onnx.defs.ONNX_DOMAIN, 13)]
    )
    onnx.checker.check_model(model_def)
    return model_def


def pytorch_inference(benchmark, ort_type: int, torch_type: torch.dtype, device: str):
    model_float = get_simple_onnx(input_type=ort_type)
    onnx_device = "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"
    ort_model = InferenceSession(model_float.SerializeToString(), providers=[onnx_device])
    binding = ort_model.io_binding()
    input = torch.rand((1, 1000)).type(torch_type).to(device)
    expected = input + 2
    results = benchmark(
        inference_onnx_binding,
        model_onnx=ort_model,
        inputs={"X": input},
        device="cpu",
        binding=binding,
        clone_tensor=False,
    )
    assert np.allclose(results["OUT"].cpu().numpy(), expected.cpu().numpy(), atol=1e-1)


@pytest.mark.benchmark(group="pytorch_inference", disable_gc=True, warmup=True)
def test_pytorch_inference_float32_cpu(benchmark):
    pytorch_inference(benchmark=benchmark, ort_type=TensorProto.FLOAT, torch_type=torch.float32, device="cpu")


@pytest.mark.benchmark(group="pytorch_inference", disable_gc=True, warmup=True)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_pytorch_inference_float32_cuda(benchmark):
    pytorch_inference(benchmark=benchmark, ort_type=TensorProto.FLOAT, torch_type=torch.float32, device="cuda")


@pytest.mark.benchmark(group="pytorch_inference", disable_gc=True, warmup=True)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_pytorch_inference_float16_cuda(benchmark):
    pytorch_inference(benchmark=benchmark, ort_type=TensorProto.FLOAT16, torch_type=torch.float16, device="cuda")


def test_merge_onnx():
    model1 = get_simple_onnx(input_type=TensorProto.FLOAT)
    model1_path = tempfile.NamedTemporaryFile()
    save_onnx(model1, model1_path.name)
    model2 = get_simple_onnx(input_type=TensorProto.FLOAT)
    model2_path = tempfile.NamedTemporaryFile()
    save_onnx(model2, model2_path.name)
    output = tempfile.NamedTemporaryFile()
    merge_autoregressive_model_graphs(model1_path.name, model2_path.name, output.name)
    ort_model = InferenceSession(output.name, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in ort_model.get_inputs()]
    assert input_names == ["X", "enable_cache"]
    output_names = [o.name for o in ort_model.get_outputs()]
    assert output_names == ["OUT"]


def get_onnx_if() -> ModelProto:
    """
    Generate an ONNX model with an If node
    :return: Onnx model
    """
    # I/O
    input_x: ValueInfoProto = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape=["axis1", "axis2"])
    input_y: ValueInfoProto = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape=["axis1", "axis2"])
    out: ValueInfoProto = helper.make_tensor_value_info("OUT", TensorProto.FLOAT, shape=[1, 4])
    output_z: ValueInfoProto = helper.make_tensor_value_info("Z", TensorProto.FLOAT, shape=[1, 4])

    # Create a node (NodeProto)
    add_1_node: NodeProto = helper.make_node(
        op_type="Add",
        inputs=["X", "Tensor"],
        outputs=["add1"],
        name="first_add",
    )

    add_2_node: NodeProto = helper.make_node(
        op_type="Add",
        inputs=["add1", "Tensor"],
        outputs=["OUT"],
        name="second_add",
    )

    graph_add: GraphProto = helper.make_graph(
        nodes=[add_1_node, add_2_node],
        name="add-model",
        inputs=[],
        outputs=[out],
        initializer=[],
    )

    # Create a node (NodeProto)
    sub_1_node: NodeProto = helper.make_node(
        op_type="Sub",
        inputs=["X", "Tensor"],
        outputs=["sub1"],
        name="first_sub",
    )

    sub_2_node: NodeProto = helper.make_node(
        op_type="Sub",
        inputs=["sub1", "Tensor"],
        outputs=["OUT"],
        name="second_sub",
    )

    graph_sub: GraphProto = helper.make_graph(
        nodes=[sub_1_node, sub_2_node],
        name="sub-model",
        inputs=[],
        outputs=[out],
        initializer=[],
    )

    retrieve_shape: NodeProto = helper.make_node(
        op_type="Shape",
        inputs=["Y"],
        outputs=["shape_2"],
        name="retrieve_shape",
    )

    gather_def: NodeProto = helper.make_node(
        op_type="Gather",
        inputs=["shape_2", "gather_dim_index"],
        outputs=["gather1"],
        name="gather_def",
    )

    equal_def: NodeProto = helper.make_node(
        op_type="Equal",
        inputs=["gather1", "expected_val"],
        outputs=["eq_vec"],
        name="equal_def",
    )

    squeeze_def: NodeProto = helper.make_node(
        op_type="Squeeze",
        inputs=["eq_vec", "squeeze_dim"],
        outputs=["if_cond"],
        name="squeeze_def",
    )

    if_node = onnx.helper.make_node(
        "If", inputs=["if_cond"], outputs=["Z"], then_branch=graph_add, else_branch=graph_sub, name="if_node"
    )

    # vals are flatten values
    init_tensor = helper.make_tensor("Tensor", dims=(1, 4), vals=[1.0, 2.0, 3.0, 4.0], data_type=TensorProto.FLOAT)
    gather_dim_index = helper.make_tensor("gather_dim_index", dims=(1,), vals=[1], data_type=TensorProto.INT32)
    expected_val = helper.make_tensor("expected_val", dims=(1,), vals=[4], data_type=TensorProto.INT64)
    squeeze_dim = helper.make_tensor("squeeze_dim", dims=(1,), vals=[0], data_type=TensorProto.INT64)

    # Create the graph (GraphProto)
    if_graph_def: GraphProto = helper.make_graph(
        nodes=[retrieve_shape, gather_def, equal_def, squeeze_def, if_node],
        name="if-model",
        inputs=[input_x, input_y],
        outputs=[output_z],
        initializer=[gather_dim_index, expected_val, squeeze_dim, init_tensor],
    )

    model_def: ModelProto = helper.make_model(
        if_graph_def, producer_name="onnx-example", opset_imports=[helper.make_opsetid(onnx.defs.ONNX_DOMAIN, 13)]
    )
    onnx.checker.check_model(model_def)
    return model_def


def test_io_mapping():
    if_model = get_onnx_if()
    inputs, outputs = get_io_to_node_mapping(onnx_model=if_model)
    assert len(inputs) == 12
    assert len(outputs) == 8
    # check k, v are all defined
    for k, v in list(inputs.items()) + list(outputs.items()):
        assert k
        assert v


def check_pytorch_conversion(data: np.ndarray, ortvalue: OrtValue, benchmark: Optional[BenchmarkFixture]):
    """
    Check that the converter to PyTorch tensor produces the same output as ONNX model
    """
    if benchmark is not None:
        tensor = benchmark(to_pytorch, ort_tensor=ortvalue, clone_tensor=False)
    else:
        tensor = to_pytorch(ort_tensor=ortvalue, clone_tensor=False)
    if ortvalue.device_name().lower() == "cuda":
        assert tensor.is_cuda
    expected_dtype, *_ = ort_conversion_table[ortvalue.data_type()]
    assert tensor.dtype == expected_dtype
    assert tensor.shape == data.shape
    assert np.allclose(tensor.cpu().numpy(), data)


@pytest.mark.benchmark(group="ort_pytorch_conversion", disable_gc=True, warmup=True)
def test_ort_conversion_cpu_float16(benchmark):
    """
    float16 doesn't exist in ctypes, it will use an intermediate dtype during the conversion,
    making the conversion slower.
    """
    data = np.arange(1000, dtype=np.float16)
    ortvalue = OrtValue.ortvalue_from_numpy(data)
    check_pytorch_conversion(data=data, ortvalue=ortvalue, benchmark=benchmark)


@pytest.mark.benchmark(group="ort_pytorch_conversion", disable_gc=True, warmup=True)
def test_ort_conversion_cpu_float32(benchmark):
    data = np.arange(1000, dtype=np.float32)
    ortvalue = OrtValue.ortvalue_from_numpy(data)
    check_pytorch_conversion(data=data, ortvalue=ortvalue, benchmark=benchmark)


@pytest.mark.benchmark(group="ort_pytorch_conversion", disable_gc=True, warmup=True)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_ort_conversion_gpu_float16(benchmark):
    data = np.arange(1000, dtype=np.float16)
    ortvalue = OrtValue.ortvalue_from_numpy(data, "cuda", 0)
    check_pytorch_conversion(data=data, ortvalue=ortvalue, benchmark=benchmark)


@pytest.mark.benchmark(group="ort_pytorch_conversion", disable_gc=True, warmup=True)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_ort_conversion_gpu_float32(benchmark):
    data = np.arange(1000, dtype=np.float32)
    ortvalue = OrtValue.ortvalue_from_numpy(data, "cuda", 0)
    check_pytorch_conversion(data=data, ortvalue=ortvalue, benchmark=benchmark)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_to_pytorch():
    content = np.arange(8).reshape(2, 4)
    test_data: List[np.ndarray] = [
        np.float16(content),
        np.float32(content),
        np.float64(content),
        np.bool_(content),
        np.int8(content),
        np.int16(content),
        np.int32(content),
        np.int64(content),
        np.array(1, dtype=np.float64),
        np.array(1, dtype=np.int64),
    ]
    for data in test_data:
        ortvalue = OrtValue.ortvalue_from_numpy(data)
        check_pytorch_conversion(data=data, ortvalue=ortvalue, benchmark=None)
        ortvalue = OrtValue.ortvalue_from_numpy(data, "cuda", 0)
        check_pytorch_conversion(data=data, ortvalue=ortvalue, benchmark=None)


def test_to_pytorch_update_ort_value_inplace():
    data = np.arange(4, dtype=np.float32).reshape((2, 2))
    ortvalue = OrtValue.ortvalue_from_numpy(data)
    tensor = to_pytorch(ort_tensor=ortvalue, clone_tensor=False)
    assert np.allclose(tensor.numpy(), data)

    new_data = data + 1
    assert not np.allclose(new_data, data)
    ortvalue.update_inplace(new_data)
    assert np.allclose(tensor.cpu().numpy(), new_data)
    # check data update doesn't impact torch tensor
    ortvalue.update_inplace(data)
    tensor = to_pytorch(ort_tensor=ortvalue, clone_tensor=True)
    assert np.allclose(tensor.numpy(), data)
    ortvalue.update_inplace(new_data)
    assert np.allclose(tensor.numpy(), data)


@pytest.mark.benchmark(group="bf16", disable_gc=True, warmup=False)
def test_conversion_to_bf16(benchmark):
    fp32_random_array = np.random.random(10).astype(np.float32)
    bf16_bytes = benchmark(convert_fp32_to_bf16, fp32_data=fp32_random_array.tobytes())
    fp32_result = torch.frombuffer(bf16_bytes, dtype=torch.bfloat16).type(torch.float32).numpy()
    assert np.allclose(fp32_result, fp32_random_array, atol=0.01)


@pytest.mark.benchmark(group="bf16", disable_gc=True, warmup=False)
def test_conversion_from_bf16(benchmark):
    original_bf16 = torch.rand(10, dtype=torch.bfloat16)
    original_as_fp32 = original_bf16.type(torch.float32).numpy()
    bf16_bytes = original_bf16.view(torch.int16).numpy().tobytes()
    conversion_fp32_bytes = benchmark(convert_bf16_to_fp32, bf16_data=bf16_bytes)
    conversion_fp32 = numpy.frombuffer(conversion_fp32_bytes, dtype=np.float32)
    assert np.allclose(original_as_fp32, conversion_fp32, atol=0.01)
