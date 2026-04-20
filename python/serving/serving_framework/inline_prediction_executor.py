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

"""A thin shell to fit a predictor function into the PredictionExecutor interface.

This is a convenience wrapper to allow a predictor function to be used directly
as a PredictionExecutor.

Intended usage in launching a server:
predictor = Predictor()
executor = InlinePredictionExecutor(predictor.predict)
server_gunicorn.PredictionApplication(executor, ...).run()
"""

from collections.abc import Callable
from typing import Any

from typing_extensions import override

from serving.serving_framework import model_runner
from serving.serving_framework import server_gunicorn


class InlinePredictionExecutor(server_gunicorn.PredictionExecutor):
  """Provides prediction request execution using an inline function.

  Provides a little framing to simplify the use of predictor functions in the
  server worker process.

  If a function call with no setup is insufficient, overriding the start and
  predict methods here can be used to provide more complex behavior. Inheritance
  directly from PredictionExecutor is also an option.
  """

  def __init__(
      self,
      predictor: Callable[
          [dict[str, Any], model_runner.ModelRunner], dict[str, Any]
      ],
      model_runner_source: Callable[[], model_runner.ModelRunner]
  ):
    self._predictor = predictor
    self._model_runner = None
    self._model_runner_source = model_runner_source

  @override
  def start(self) -> None:
    """Starts the executor.

    Called after the Gunicorn worker process has started. Performs any setup
    which needs to be done post-fork.
    """
    # Safer to instantiate the RPC stub post-fork.
    self._model_runner = self._model_runner_source()

  def predict(self, input_json: dict[str, Any]) -> dict[str, Any]:
    """Executes the given request payload."""
    if self._model_runner is None:
      raise RuntimeError(
          "Model runner is not initialized. Please call start() first."
      )
    return self._predictor(input_json, self._model_runner)

  @override
  def execute(self, input_json: dict[str, Any]) -> dict[str, Any]:
    """Executes the given request payload.

    Args:
      input_json: The full json prediction request payload.

    Returns:
      The json response to the prediction request.

    Raises:
      RuntimeError: Prediction failed in an unhandled way.
    """
    try:
      return self.predict(input_json)
    except Exception as e:
      raise RuntimeError("Unhandled exception from predictor.") from e
