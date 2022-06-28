#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import importlib
import inspect
import logging
from typing import Any, Dict


def code_utils(
    module_name: str, function: Any, new_function_name: str, modifications: Dict[str, str], log: logging.Logger
) -> bool:
    """
    This function helps updating a module given the function name and the modifications to be done on this function
    Once you use update_module(), you just need to override the function with its new version using the new function
    name. If you want to replace the whole code function you can pass * in src code as a key.
    :param module_name: the module to be updated
    :param function: the function to be updated in the given module
    :param modifications: a dictionary containing all the modifications to be done, keys are the source/original code
    return: Whether it succeeded to update the given function
    and values are the new code to be used to replace source code
    Example:
    if you're updating the forward function in T5Attention transformers
    `transformers.models.t5.modeling_t5.T5Attention.forward` and using `updatedForward` as new function name, you can
    do:
        >>> import transformers
        >>> code_utils(module_name="transformers.models.t5.modeling_t5",
        >>>               function=transformers.models.t5.modeling_t5.T5Attention.forward ,
        >>>               new_function_name="updatedForward",
        >>>               modifications=dict("*", "return True"))
        >>> transformers.models.t5.modeling_t5.T5Attention.forward = transformers.models.t5.modeling_t5.updatedForward
    """
    try:
        model_module = importlib.import_module(name=module_name)
        function_code = inspect.getsource(function)
        for src_code, new_code in modifications.items():
            function_code = function_code.replace("):\n", "\n):\n")  # newline to align the code after function header
            function_code = inspect.cleandoc(function_code)
            if src_code == "*":
                # find the right indentation used with the function header to add it for the new code
                # here it is specific for replacing the whole function body
                header_end_idx = function_code.find("):\n")
                function_code = function_code[: header_end_idx + len("):\n")] + f"    {new_code}"
            else:
                function_code = function_code.replace(src_code, new_code)
        function_code = function_code.replace(f"def {function.__name__}", f"def {new_function_name}")
        exec(inspect.cleandoc(function_code), model_module.__dict__, model_module.__dict__)
        return True
    except Exception as e:
        log.error(f"Failed to update the given function: {e}")
        return False
