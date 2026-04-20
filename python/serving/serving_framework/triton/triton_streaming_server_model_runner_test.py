# Copyright 2026 Google LLC
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

import asyncio
from typing import Any, Iterable
from unittest import mock

import numpy as np
import numpy.testing as npt
from tritonclient import grpc as triton_grpc
from tritonclient.grpc import aio as triton_aio

from absl.testing import absltest
from serving.serving_framework import model_runner
from serving.serving_framework.triton import triton_streaming_server_model_runner


async def async_iterator(source: Iterable[Any]) -> Any:
  for value in source:
    yield value


class TritonStreamingServerModelRunnerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.create_autospec(
        triton_aio.InferenceServerClient, instance=True
    )
    self.mock_client.__aenter__.return_value = self.mock_client
    self.mock_client_class = self.enter_context(
        mock.patch.object(triton_aio, "InferenceServerClient", autospec=True)
    )
    self.mock_client_class.return_value = self.mock_client
    self.model_runner = (
        triton_streaming_server_model_runner.TritonStreamingServerModelRunner()
    )

  def test_end_to_end(self):
    mock_response = mock.create_autospec(triton_grpc.InferResult, instance=True)
    output0 = np.array([4.0, 5.0, 6.0])
    output1 = np.array([7.0, 8.0, 9.0])

    def as_numpy_side_effect(key):
      if key == "output_0":
        return output0
      if key == "output_1":
        return output1
      return None

    mock_response.as_numpy.side_effect = as_numpy_side_effect
    self.mock_client.stream_infer.return_value = async_iterator((
        (mock_response, None),
    ))

    result = self.model_runner.run_model_multiple_output(
        model_input=np.array([1, 2, 3]),
        model_name="test_model",
        model_version=1,
        model_output_keys={"output_0", "output_1"},
        parameters={"foo": "bar"},
    )

    self.mock_client_class.assert_called_once_with(
        triton_streaming_server_model_runner._HOSTPORT
    )
    self.mock_client.stream_infer.assert_called_once()
    args, _ = self.mock_client.stream_infer.call_args
    request_generator = args[0]

    async def get_requests(g):
      return [req async for req in g]

    requests = asyncio.run(get_requests(request_generator))
    self.assertLen(requests, 1)
    request = requests[0]
    self.assertEqual(request["model_name"], "test_model")
    self.assertEqual(request["model_version"], "1")
    self.assertEqual(request["parameters"], {"foo": "bar"})
    self.assertLen(request["inputs"], 1)
    self.assertEqual(request["inputs"][0].name(), "inputs")
    # Not checking the values of the input further due to complex packing into
    # tritonclient.grpc.InferInput.
    npt.assert_array_equal(result["output_0"], output0)
    npt.assert_array_equal(result["output_1"], output1)

  def test_model_server_error(self):
    self.mock_client.stream_infer.return_value = async_iterator((
        (None, triton_grpc.InferenceServerException("test error")),
    ))
    with self.assertRaisesRegex(
        model_runner.ModelError, "Error running model: test error"
    ):
      self.model_runner.run_model_multiple_output(
          model_input=np.array([1, 2, 3]),
          model_name="test_model",
          model_version=1,
          model_output_keys={"output_0", "output_1"},
          parameters={"foo": "bar"},
      )

  def test_model_server_cancelled(self):
    self.mock_client.stream_infer.return_value = async_iterator((
        (None, asyncio.CancelledError("test error")),
    ))
    with self.assertRaisesRegex(
        RuntimeError,
        "Model server request was cancelled. This may be due to the server"
        " shutting down or an internal asyncio error.",
    ):
      self.model_runner.run_model_multiple_output(
          model_input=np.array([1, 2, 3]),
          model_name="test_model",
          model_version=1,
          model_output_keys={"output_0", "output_1"},
          parameters={"foo": "bar"},
      )

  def test_model_server_no_result(self):
    self.mock_client.stream_infer.return_value = async_iterator(())
    with self.assertRaisesRegex(
        RuntimeError,
        "Stream from model server closed without yielding a result.",
    ):
      self.model_runner.run_model_multiple_output(
          model_input=np.array([1, 2, 3]),
          model_name="test_model",
          model_version=1,
          model_output_keys={"output_0", "output_1"},
          parameters={"foo": "bar"},
      )

  def test_model_server_missing_output_key(self):
    mock_response = mock.create_autospec(triton_grpc.InferResult, instance=True)
    output0 = np.array([4.0, 5.0, 6.0])

    def as_numpy_side_effect(key):
      if key == "output_0":
        return output0
      return None

    mock_response.as_numpy.side_effect = as_numpy_side_effect
    self.mock_client.stream_infer.return_value = async_iterator((
        (mock_response, None),
    ))
    with self.assertRaisesRegex(
        KeyError,
        "Model output keys {'output_1'} not found in model output.",
    ):
      self.model_runner.run_model_multiple_output(
          model_input=np.array([1, 2, 3]),
          model_name="test_model",
          model_version=1,
          model_output_keys={"output_0", "output_1"},
          parameters={"foo": "bar"},
      )

  def test_input_map_handling(self):
    mock_response = mock.create_autospec(triton_grpc.InferResult, instance=True)
    output0 = np.array([4.0, 5.0, 6.0])

    def as_numpy_side_effect(key):
      if key == "output_0":
        return output0
      return None

    mock_response.as_numpy.side_effect = as_numpy_side_effect
    self.mock_client.stream_infer.return_value = async_iterator((
        (mock_response, None),
    ))

    _ = self.model_runner.run_model_multiple_output(
        model_input={
            "input_0": np.array([1, 2, 3]),
            "input_1": np.array([4, 5, 6]),
        },
        model_name="test_model",
        model_version=1,
        model_output_keys={"output_0"},
        parameters={"foo": "bar"},
    )

    self.mock_client.stream_infer.assert_called_once()
    args, _ = self.mock_client.stream_infer.call_args
    request_generator = args[0]

    async def get_requests(g):
      return [req async for req in g]

    requests = asyncio.run(get_requests(request_generator))
    self.assertLen(requests, 1)
    request = requests[0]
    self.assertLen(request["inputs"], 2)
    input_names = [input_.name() for input_ in request["inputs"]]
    self.assertCountEqual(input_names, ["input_0", "input_1"])


if __name__ == "__main__":
  absltest.main()
