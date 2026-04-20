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

"""Gunicorn application for passing requests through to the executor command.

Provides a thin, subject-agnostic request server for Vertex endpoints which
handles requests by piping their JSON bodies to the given executor command
and returning the json output.
"""

import abc
from collections.abc import Mapping, Sequence
import http
import json
import os
import subprocess
from typing import Any, Optional

from absl import logging
import flask
from gunicorn.app import base as gunicorn_base
import jsonschema
from typing_extensions import override


INSTANCES_KEY = "instances"


class PredictionExecutor(abc.ABC):
  """Wraps arbitrary implementation of executing a prediction request."""

  @abc.abstractmethod
  def execute(self, input_json: dict[str, Any]) -> dict[str, Any]:
    """Executes the given request payload."""

  def start(self) -> None:
    """Starts the executor.

    Called after the Gunicorn worker process has started. Performs any setup
    which needs to be done post-fork.
    """


class SubprocessPredictionExecutor(PredictionExecutor):
  """Provides prediction request execution using a persistent worker subprocess."""

  def __init__(self, executor_command: Sequence[str]):
    """Initializes the executor with a command to start the subprocess."""
    self._executor_command = executor_command
    self._executor_process = None

  def _restart(self) -> None:
    if self._executor_process is None:
      raise RuntimeError("Executor process not started.")

    self._executor_process.terminate()
    self.start()

  @override
  def start(self):
    """Starts the executor process."""
    self._executor_process = subprocess.Popen(
        args=self._executor_command,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )

  @override
  def execute(self, input_json: dict[str, Any]) -> dict[str, Any]:
    """Uses the executor process to execute a request.

    Args:
      input_json: The full json prediction request payload.

    Returns:
      The json response to the prediction request.

    Raises:
      RuntimeError: Executor is not started or error communicating with the
      subprocess.
    """
    if self._executor_process is None:
      raise RuntimeError("Executor process not started.")

    # Ensure json string is safe to pass through the pipe protocol.
    input_str = json.dumps(input_json).replace("\n", "")

    try:
      self._executor_process.stdin.write(input_str.encode("utf-8") + b"\n")
      self._executor_process.stdin.flush()
    except BrokenPipeError as e:
      self._restart()
      raise RuntimeError("Executor process input stream closed.") from e
    exec_result = self._executor_process.stdout.readline()
    if not exec_result:
      self._restart()
      raise RuntimeError("Executor process output stream closed.")
    try:
      return json.loads(exec_result)
    except json.JSONDecodeError as e:
      raise RuntimeError("Executor process output not valid json.") from e


class ModelServerHealthCheck(abc.ABC):
  """Checks the health of the local model server."""

  @abc.abstractmethod
  def check_health(self) -> bool:
    """Check the health of the local model server immediately."""


class _PredictRoute:
  """A callable for handling a prediction route."""

  def __init__(
      self,
      executor: PredictionExecutor,
      *,
      input_validator: jsonschema.Draft202012Validator | None = None,
      instance_input: bool = True,
      prediction_validator: jsonschema.Draft202012Validator | None = None,
  ):
    self._executor = executor
    self._input_validator = input_validator
    self._instance_input = instance_input
    self._prediction_validator = prediction_validator

  def __call__(self) -> tuple[dict[str, Any], int]:
    logging.info("predict route hit")
    json_body = flask.request.get_json(silent=True)
    if json_body is None:
      return {"error": "No JSON body."}, http.HTTPStatus.BAD_REQUEST.value
    if self._instance_input and INSTANCES_KEY not in json_body:
      return {
          "error": "No instances field in request."
      }, http.HTTPStatus.BAD_REQUEST.value

    if self._input_validator is not None:
      try:
        if self._instance_input:
          for instance in json_body[INSTANCES_KEY]:
            self._input_validator.validate(instance)
        else:
          self._input_validator.validate(json_body)
      except jsonschema.exceptions.ValidationError as e:
        logging.warning("Input validation failed")
        return {"error": str(e)}, http.HTTPStatus.BAD_REQUEST.value

    logging.debug("Dispatching request to executor.")
    try:
      exec_result = self._executor.execute(json_body)
      logging.debug("Executor returned results.")
      if self._prediction_validator is not None:
        try:
          if "predictions" in exec_result:
            for result in exec_result["predictions"]:
              self._prediction_validator.validate(result)
        except jsonschema.exceptions.ValidationError as e:
          logging.exception("Response validation failed")
          return {
              "error": "Internal server error."
          }, http.HTTPStatus.INTERNAL_SERVER_ERROR.value

      return (exec_result, http.HTTPStatus.OK.value)
    except RuntimeError:
      logging.exception("Internal error handling request: Executor failed.")
      return {
          "error": "Internal server error."
      }, http.HTTPStatus.INTERNAL_SERVER_ERROR.value


def _create_app(
    executor: PredictionExecutor,
    health_check: ModelServerHealthCheck | None,
    *,
    input_validator: jsonschema.Draft202012Validator | None = None,
    instance_input: bool = True,
    prediction_validator: jsonschema.Draft202012Validator | None = None,
    additional_routes: Mapping[str, PredictionExecutor] | None = None,
) -> flask.Flask:
  """Creates a Flask app with the given executor."""
  flask_app = flask.Flask(__name__)

  if (
      "AIP_HEALTH_ROUTE" not in os.environ
      or "AIP_PREDICT_ROUTE" not in os.environ
  ):
    raise ValueError(
        "Both of the environment variables AIP_HEALTH_ROUTE and "
        "AIP_PREDICT_ROUTE need to be specified."
    )

  def health_route() -> tuple[str, int]:
    logging.info("health route hit")
    if health_check is not None and not health_check.check_health():
      return "not ok", http.HTTPStatus.SERVICE_UNAVAILABLE.value
    return "ok", http.HTTPStatus.OK.value

  health_path = os.environ.get("AIP_HEALTH_ROUTE")
  logging.info("health path: %s", health_path)
  flask_app.add_url_rule(health_path, view_func=health_route)

  predict_route = os.environ.get("AIP_PREDICT_ROUTE")
  logging.info("predict route: %s", predict_route)
  flask_app.add_url_rule(
      rule=predict_route,
      endpoint=predict_route,
      view_func=_PredictRoute(
          executor,
          input_validator=input_validator,
          instance_input=instance_input,
          prediction_validator=prediction_validator,
      ),
      methods=["POST"],
  )

  if additional_routes:
    for route, executor in additional_routes.items():
      logging.info("additional route: %s", route)
      flask_app.add_url_rule(
          rule=route,
          endpoint=route,
          view_func=_PredictRoute(
              executor,
              instance_input=False,
          ),
          methods=["POST"],
      )

  flask_app.config["TRAP_BAD_REQUEST_ERRORS"] = True

  return flask_app


class PredictionApplication(gunicorn_base.BaseApplication):
  """Application to serve predictors on Vertex endpoints using gunicorn."""

  def __init__(
      self,
      executor: PredictionExecutor,
      *,
      health_check: ModelServerHealthCheck | None,
      options: Optional[Mapping[str, Any]] = None,
      input_validator: jsonschema.Draft202012Validator | None = None,
      instance_input: bool = True,
      prediction_validator: jsonschema.Draft202012Validator | None = None,
      additional_routes: Mapping[str, PredictionExecutor] | None = None,
  ):
    """Initializes the application.

    Args:
      executor: The executor to use for handling prediction requests.
      health_check: The health check to use for the health check route.
      options: Gunicorn application options.
      input_validator: A jsonschema validator to apply to the input instances.
      instance_input: Whether the input will contain a list of instances which
        can be validated individually if a validator is provided. If False, the
        request object structure will not be assumed and the validator will
        be applied to the entire request body.
      prediction_validator: A jsonschema validator to apply to the predictions.
      additional_routes: Additional routes to serve on the server, keyed by
        route path. Validation is not applied to these routes.
    """
    self.options = options or {}
    self.options = dict(self.options)
    self.options["preload_app"] = False
    self._executor = executor
    self.application = _create_app(
        self._executor,
        health_check,
        input_validator=input_validator,
        instance_input=instance_input,
        prediction_validator=prediction_validator,
        additional_routes=additional_routes,
    )
    super().__init__()

  def load_config(self):
    config = {
        key: value
        for key, value in self.options.items()
        if key in self.cfg.settings and value is not None
    }
    for key, value in config.items():
      self.cfg.set(key.lower(), value)

  def load(self) -> flask.Flask:
    self._executor.start()
    return self.application
