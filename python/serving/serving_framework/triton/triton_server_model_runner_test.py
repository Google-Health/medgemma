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

from unittest import mock

import numpy as np
from tritonclient import grpc as triton_grpc

from absl.testing import absltest
from serving.serving_framework.triton import triton_server_model_runner
from tritonclient.grpc import service_pb2


class TritonServerModelRunnerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self._client = mock.create_autospec(
        triton_grpc.InferenceServerClient, instance=True
    )
    self._client.infer = mock.MagicMock()
    self._runner = triton_server_model_runner.TritonServerModelRunner(
        client=self._client
    )

    self._output_proto = service_pb2.ModelInferResponse(
        model_name="test_model",
        model_version="1",
        outputs=[
            service_pb2.ModelInferResponse.InferOutputTensor(
                name="output_a",
                shape=[2, 2],
                datatype="FP32",
            ),
            service_pb2.ModelInferResponse.InferOutputTensor(
                name="output_b",
                shape=[1, 2],
                datatype="INT64",
            ),
            service_pb2.ModelInferResponse.InferOutputTensor(
                name="output_c",
                shape=[3, 2],
                datatype="FP32",
            ),
        ],
        raw_output_contents=[
            np.ones((2, 2), dtype=np.float32).tobytes(),
            np.array([[7] * 2], dtype=np.int64).tobytes(),
            np.zeros((3, 2), dtype=np.float32).tobytes(),
        ],
    )

  def test_run_map_check_input(self):
    """Tests that an input map is passed to the model correctly."""
    input_map = {
        "a": np.array([[0.5] * 3] * 3, dtype=np.float32),
        "b": np.array([[2] * 2] * 3, dtype=np.int64),
    }
    self._client.infer.return_value = triton_grpc.InferResult(
        self._output_proto
    )

    _ = self._runner.run_model_multiple_output(
        input_map,
        model_name="test_model",
        model_output_keys={"output_a", "output_b", "output_c"},
    )

    self.assertLen(self._client.infer.call_args_list, 1)
    self.assertEqual(
        self._client.infer.call_args[0][0],
        "test_model",
        "Model name passed to model does not match expectation.",
    )
    self.assertEqual(
        self._client.infer.call_args[0][2],
        "",
        "Model version passed to model does not match expectation.",
    )
    self.assertEqual(
        self._client.infer.call_args[0][1][0]._get_tensor(),
        service_pb2.ModelInferRequest.InferInputTensor(
            name="a",
            shape=[3, 3],
            datatype="FP32",
        ),
        msg="Input tensor passed to model does not match expectation.",
    )
    self.assertEqual(
        self._client.infer.call_args[0][1][1]._get_tensor(),
        service_pb2.ModelInferRequest.InferInputTensor(
            name="b",
            shape=[3, 2],
            datatype="INT64",
        ),
        msg="Input tensor passed to model does not match expectation.",
    )

  def test_run_bare_check_input(self):
    """Tests the handling of a bare input tensor  passed to the model."""
    input_tensor = np.array([[0.5] * 3] * 3, dtype=np.float32)
    self._client.infer.return_value = triton_grpc.InferResult(
        self._output_proto
    )

    _ = self._runner.run_model_multiple_output(
        input_tensor,
        model_name="test_model",
        model_output_keys={"output_a", "output_b", "output_c"},
    )

    self.assertLen(self._client.infer.call_args_list, 1)
    self.assertEqual(
        self._client.infer.call_args[0][0],
        "test_model",
        "Model name passed to model does not match expectation.",
    )
    self.assertEqual(
        self._client.infer.call_args[0][2],
        "",
        "Model version passed to model does not match expectation.",
    )
    self.assertEqual(
        self._client.infer.call_args[0][1][0]._get_tensor(),
        service_pb2.ModelInferRequest.InferInputTensor(
            name="inputs",
            shape=[3, 3],
            datatype="FP32",
        ),
        msg="Input tensor passed to model does not match expectation.",
    )

  def test_input_model_version(self):
    """Tests that the model version is passed to the model correctly."""
    input_tensor = np.array([[0.5] * 3] * 3, dtype=np.float32)
    self._client.infer.return_value = triton_grpc.InferResult(
        self._output_proto
    )

    _ = self._runner.run_model_multiple_output(
        input_tensor,
        model_name="test_model",
        model_version=3,
        model_output_keys={"output_a", "output_b", "output_c"},
    )

    self.assertEqual(
        self._client.infer.call_args[0][2],
        "3",
        "Model version passed to model does not match expectation.",
    )

  def test_run_check_output(self):
    """Tests that the output is returned correctly."""
    input_map = {
        "a": np.array([[0.5] * 3] * 3, dtype=np.float32),
        "b": np.array([[2] * 2] * 3, dtype=np.int64),
    }
    self._client.infer.return_value = triton_grpc.InferResult(
        self._output_proto
    )

    result = self._runner.run_model_multiple_output(
        input_map,
        model_name="test_model",
        model_output_keys={"output_a", "output_b"},
    )

    np.testing.assert_array_equal(
        result["output_a"],
        np.ones((2, 2), dtype=np.float32),
        "Output tensor passed to model does not match expectation.",
    )
    np.testing.assert_array_equal(
        result["output_b"],
        np.array([[7] * 2], dtype=np.int64),
        "Output tensor passed to model does not match expectation.",
    )
    self.assertLen(result, 2)

  def test_run_check_missing_key(self):
    """Tests that the output is returned correctly."""
    input_map = {
        "a": np.array([[0.5] * 3] * 3, dtype=np.float32),
        "b": np.array([[2] * 2] * 3, dtype=np.int64),
    }
    self._client.infer.return_value = triton_grpc.InferResult(
        self._output_proto
    )

    with self.assertRaises(KeyError):
      _ = self._runner.run_model_multiple_output(
          input_map,
          model_name="test_model",
          model_output_keys={"output_a", "output_d"},
      )

if __name__ == "__main__":
  absltest.main()
