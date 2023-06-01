from typing import List, Union

import numpy as np
import torch

from transformer_deploy.benchmarks.utils import compare_outputs, to_numpy


def check_accuracy(
    engine_name: str,
    pytorch_output: List[torch.Tensor],
    engine_output: List[Union[np.ndarray, torch.Tensor]],
    tolerance: float,
) -> None:
    """
    Compare engine predictions with a reference.
    Assert that the difference is under a threshold.

    :param engine_name: string used in error message, if any
    :param pytorch_output: reference output used for the comparaison
    :param engine_output: output from the engine
    :param tolerance: if difference in outputs is above threshold, an error will be raised
    """
    pytorch_output = to_numpy(pytorch_output)
    engine_output = to_numpy(engine_output)
    discrepency = compare_outputs(pytorch_output=pytorch_output, engine_output=engine_output)
    assert discrepency <= tolerance, (
        f"{engine_name} discrepency is too high ({discrepency:.2f} >= {tolerance}):\n"
        f"Pythorch:\n{pytorch_output}\n"
        f"VS\n"
        f"Engine:\n{engine_output}\n"
        f"Diff:\n"
        f"{torch.asarray(pytorch_output) - torch.asarray(engine_output)}\n"
        "Tolerance can be increased with --atol parameter."
    )
