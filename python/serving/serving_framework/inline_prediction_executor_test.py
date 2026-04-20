# Copyright 2024 Google LLC
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

from collections.abc import Mapping, Set
from unittest import mock
from typing import Any

import numpy as np

from absl.testing import absltest
from serving.serving_framework import inline_prediction_executor
from serving.serving_framework import model_runner


class DummyModelRunner(model_runner.ModelRunner):
  """Dummy model runner for testing."""

  def run_model_multiple_output(
      self,
      model_input: Mapping[str, np.ndarray] | np.ndarray,
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_keys: Set[str],
      parameters: Mapping[str, Any] | None = None,
  ) -> Mapping[str, np.ndarray]:
    del model_name, model_version, model_output_keys, parameters
    return {"output_0": np.ones((1, 2), dtype=np.float32)}


class InlinePredictionExecutorTest(absltest.TestCase):

  def test_predict_requires_start(self):
    predictor = mock.MagicMock()
    executor = inline_prediction_executor.InlinePredictionExecutor(
        predictor, DummyModelRunner
    )
    with self.assertRaises(RuntimeError):
      executor.predict({"placeholder": "input"})

  def test_execute_catches_predictor_exception(self):
    predictor = mock.MagicMock(side_effect=Exception("test error"))
    executor = inline_prediction_executor.InlinePredictionExecutor(
        predictor, DummyModelRunner
    )
    executor.start()
    with self.assertRaises(RuntimeError):
      executor.execute({"placeholder": "input"})

  def test_execute_calls_predictor(self):
    predictor = mock.MagicMock(return_value={"placeholder": "output"})
    mock_model_runner = mock.create_autospec(
        DummyModelRunner, instance=True
    )
    mock_model_runner_class = mock.create_autospec(
        DummyModelRunner, autospec=True
    )
    mock_model_runner_class.return_value = mock_model_runner
    executor = inline_prediction_executor.InlinePredictionExecutor(
        predictor, mock_model_runner_class
    )

    executor.start()
    self.assertEqual(
        executor.execute({"placeholder": "input"}),
        {"placeholder": "output"},
    )
    mock_model_runner_class.assert_called_once()
    predictor.assert_called_once_with(
        {"placeholder": "input"}, mock_model_runner
    )


if __name__ == "__main__":
  absltest.main()
