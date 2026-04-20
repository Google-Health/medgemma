# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ModelRunner implementation using grpc to NVIDIA Triton model server.

Uses Triton GRPC client and relies on a model server running locally.
"""

from collections.abc import Mapping, Set
from typing import Any

from absl import logging
import numpy as np
from tritonclient import grpc as triton_grpc
from tritonclient import utils as triton_utils
from typing_extensions import override

from serving.serving_framework import model_runner

_HOSTPORT = "localhost:8500"


class TritonServerModelRunner(model_runner.ModelRunner):
  """ModelRunner implementation using grpc to NVIDIA Triton model server."""

  def __init__(self, client: triton_grpc.InferenceServerClient | None = None):
    if client is not None:
      self._client = client
    else:
      self._client = triton_grpc.InferenceServerClient(_HOSTPORT)

  @override
  def run_model_multiple_output(
      self,
      model_input: Mapping[str, np.ndarray] | np.ndarray,
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_keys: Set[str],
      parameters: Mapping[str, Any] | None = None,
  ) -> Mapping[str, np.ndarray]:
    """Runs a model on the given input and returns multiple outputs.

    Args:
      model_input: An array or map of arrays comprising the input tensors for
        the model. A bare array is keyed by "inputs".
      model_name: The name of the model to run.
      model_version: The version of the model to run. Uses default if None.
      model_output_keys: The desired model output keys.
      parameters: Additional parameters to pass to the model.

    Returns:
      A mapping of model output keys to tensors.

    Raises:
      KeyError: If any of the model_output_keys are not found in the model
        output.
    """
    # If a bare np.ndarray was passed, it will be passed using the default
    # input key "inputs".
    # If a Mapping was passed, use the keys from that mapping.
    if isinstance(model_input, np.ndarray):
      logging.debug("Handling bare input tensor.")
      input_map = {"inputs": model_input}
    else:
      input_map = model_input

    model_version = str(model_version) if model_version is not None else ""

    inputs = []
    for key, data in input_map.items():
      input_tensor = triton_grpc.InferInput(
          key, data.shape, triton_utils.np_to_triton_dtype(data.dtype)
      )
      input_tensor.set_data_from_numpy(data)
      inputs.append(input_tensor)

    model_parameters = None if parameters is None else dict(parameters)

    result = self._client.infer(
        model_name, inputs, model_version, parameters=model_parameters
    )
    assert result is not None  # infer never returns None, despite annotation.

    outputs = {key: result.as_numpy(key) for key in model_output_keys}
    missing_keys = {key for key in model_output_keys if outputs[key] is None}
    if missing_keys:
      raise KeyError(
          f"Model output keys {missing_keys} not found in model output."
      )
    return outputs
