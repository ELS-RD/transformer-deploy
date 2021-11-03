# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import struct

TRITON_STRING_TO_NUMPY = {
    'TYPE_BOOL': bool,
    'TYPE_UINT8': np.uint8,
    'TYPE_UINT16': np.uint16,
    'TYPE_UINT32': np.uint32,
    'TYPE_UINT64': np.uint64,
    'TYPE_INT8': np.int8,
    'TYPE_INT16': np.int16,
    'TYPE_INT32': np.int32,
    'TYPE_INT64': np.int64,
    'TYPE_FP16': np.float16,
    'TYPE_FP32': np.float32,
    'TYPE_FP64': np.float64,
    'TYPE_STRING': np.object_
}


def serialize_byte_tensor(input_tensor):
    """
    Serializes a bytes tensor into a flat numpy array of length prepended
    bytes. The numpy array should use dtype of np.object_. For np.bytes_,
    numpy will remove trailing zeros at the end of byte sequence and because
    of this it should be avoided.
    Parameters
    ----------
    input_tensor : np.array
        The bytes tensor to serialize.
    Returns
    -------
    serialized_bytes_tensor : np.array
        The 1-D numpy array of type uint8 containing the serialized bytes in 'C' order.
    Raises
    ------
    InferenceServerException
        If unable to serialize the given tensor.
    """

    if input_tensor.size == 0:
        return ()

    # If the input is a tensor of string/bytes objects, then must flatten those
    # into a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in "C" order.
    if (input_tensor.dtype == np.object_) or (input_tensor.dtype.type
                                              == np.bytes_):
        flattened_ls = []
        for obj in np.nditer(input_tensor, flags=["refs_ok"], order='C'):
            # If directly passing bytes to BYTES type,
            # don't convert it to str as Python will encode the
            # bytes which may distort the meaning
            if input_tensor.dtype == np.object_:
                if type(obj.item()) == bytes:
                    s = obj.item()
                else:
                    s = str(obj.item()).encode('utf-8')
            else:
                s = obj.item()
            flattened_ls.append(struct.pack("<I", len(s)))
            flattened_ls.append(s)
        flattened = b''.join(flattened_ls)
        return flattened
    return None


def deserialize_bytes_tensor(encoded_tensor):
    """
    Deserializes an encoded bytes tensor into an
    numpy array of dtype of python objects
    Parameters
    ----------
    encoded_tensor : bytes
        The encoded bytes tensor where each element
        has its length in first 4 bytes followed by
        the content
    Returns
    -------
    string_tensor : np.array
        The 1-D numpy array of type object containing the
        deserialized bytes in 'C' order.
    """
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return (np.array(strs, dtype=np.object_))


def get_input_tensor_by_name(inference_request, name):
    """Find an input Tensor in the inference_request that has the given
    name
    Parameters
    ----------
    inference_request : InferenceRequest
        InferenceRequest object
    name : str
        name of the input Tensor object
    Returns
    -------
    Tensor
        The input Tensor with the specified name, or None if no
        input Tensor with this name exists
    """
    input_tensors = inference_request.inputs()
    for input_tensor in input_tensors:
        if input_tensor.name() == name:
            return input_tensor

    return None


def get_output_tensor_by_name(inference_response, name):
    """Find an output Tensor in the inference_response that has the given
    name
    Parameters
    ----------
    inference_response : InferenceResponse
        InferenceResponse object
    name : str
        name of the output Tensor object
    Returns
    -------
    Tensor
        The output Tensor with the specified name, or None if no
        output Tensor with this name exists
    """
    output_tensors = inference_response.output_tensors()
    for output_tensor in output_tensors:
        if output_tensor.name() == name:
            return output_tensor

    return None


def get_input_config_by_name(model_config, name):
    """Get input properties corresponding to the input
    with given `name`
    Parameters
    ----------
    model_config : dict
        dictionary object containing the model configuration
    name : str
        name of the input object
    Returns
    -------
    dict
        A dictionary containing all the properties for a given input
        name, or None if no input with this name exists
    """
    if 'input' in model_config:
        inputs = model_config['input']
        for input_properties in inputs:
            if input_properties['name'] == name:
                return input_properties

    return None


def get_output_config_by_name(model_config, name):
    """Get output properties corresponding to the output
    with given `name`
    Parameters
    ----------
    model_config : dict
        dictionary object containing the model configuration
    name : str
        name of the output object
    Returns
    -------
    dict
        A dictionary containing all the properties for a given output
        name, or None if no output with this name exists
    """
    if 'output' in model_config:
        outputs = model_config['output']
        for output_properties in outputs:
            if output_properties['name'] == name:
                return output_properties

    return None


def triton_to_numpy_type(data_type):
    if data_type == 1:
        return np.bool_
    elif data_type == 2:
        return np.uint8
    elif data_type == 3:
        return np.uint16
    elif data_type == 4:
        return np.uint32
    elif data_type == 5:
        return np.uint64
    elif data_type == 6:
        return np.int8
    elif data_type == 7:
        return np.int16
    elif data_type == 8:
        return np.int32
    elif data_type == 9:
        return np.int64
    elif data_type == 10:
        return np.float16
    elif data_type == 11:
        return np.float32
    elif data_type == 12:
        return np.float64
    elif data_type == 13:
        return np.object_


def numpy_to_triton_type(data_type):
    if data_type == np.bool_:
        return 1
    elif data_type == np.uint8:
        return 2
    elif data_type == np.uint16:
        return 3
    elif data_type == np.uint32:
        return 4
    elif data_type == np.uint64:
        return 5
    elif data_type == np.int8:
        return 6
    elif data_type == np.int16:
        return 7
    elif data_type == np.int32:
        return 8
    elif data_type == np.int64:
        return 9
    elif data_type == np.float16:
        return 10
    elif data_type == np.float32:
        return 11
    elif data_type == np.float64:
        return 12
    elif data_type == np.object_ or data_type == np.bytes_:
        return 13


def triton_string_to_numpy(triton_type_string):
    return TRITON_STRING_TO_NUMPY[triton_type_string]
