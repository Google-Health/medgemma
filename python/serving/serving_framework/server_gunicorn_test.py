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

import http
import io
import os
import subprocess
from unittest import mock

from absl.testing import absltest
from serving.serving_framework import server_gunicorn


class DummyHealthCheck:

  def __init__(self, check_result: bool):
    self._check_result = check_result

  def check_health(self):
    return self._check_result


class ServerGunicornTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["AIP_PREDICT_ROUTE"] = "/fake-predict-route"
    os.environ["AIP_HEALTH_ROUTE"] = "/fake-health-route"

  def test_application_option_default(self):
    executor = mock.create_autospec(
        server_gunicorn.PredictionExecutor,
        instance=True,
    )

    app = server_gunicorn.PredictionApplication(executor, health_check=None)

    self.assertEqual(app.cfg.workers, 1)

  def test_application_option_setting(self):
    options = {
        "workers": 3,
    }
    executor = mock.create_autospec(
        server_gunicorn.PredictionExecutor,
        instance=True,
    )

    app = server_gunicorn.PredictionApplication(
        executor, options=options, health_check=None
    )

    self.assertEqual(app.cfg.workers, 3)

  def test_health_route_no_check(self):

    executor = mock.create_autospec(
        server_gunicorn.PredictionExecutor,
        instance=True,
    )

    app = server_gunicorn.PredictionApplication(
        executor, health_check=None
    ).load()
    service = app.test_client()

    response = service.get("/fake-health-route")

    self.assertEqual(response.status_code, http.HTTPStatus.OK)
    self.assertEqual(response.text, "ok")

  def test_predict_route_no_json(self):
    executor = mock.create_autospec(
        server_gunicorn.PredictionExecutor,
        instance=True,
    )
    app = server_gunicorn.PredictionApplication(
        executor, health_check=None
    ).load()
    service = app.test_client()

    response = service.post("/fake-predict-route", data="invalid")

    executor.start.assert_called_once()
    executor.execute.assert_not_called()
    self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)
    self.assertDictEqual({"error": "No JSON body."}, response.get_json())

  def test_predict_route(self):
    executor = mock.create_autospec(
        server_gunicorn.PredictionExecutor,
        instance=True,
    )
    app = server_gunicorn.PredictionApplication(
        executor, health_check=None
    ).load()
    service = app.test_client()
    executor.execute.return_value = {"placeholder": "output"}

    response = service.post(
        "/fake-predict-route", json={"instances": [{"filler": "filler"}]}
    )

    executor.start.assert_called_once()
    executor.execute.assert_called_once_with(
        {"instances": [{"filler": "filler"}]}
    )
    self.assertEqual(response.status_code, http.HTTPStatus.OK)
    self.assertDictEqual({"placeholder": "output"}, response.get_json())

  def test_subprocess_executor_execute(self):
    mock_process = mock.create_autospec(subprocess.Popen, instance=True)
    with mock.patch.object(
        subprocess, "Popen", autospec=True, return_value=mock_process
    ) as mock_popen:
      executor = server_gunicorn.SubprocessPredictionExecutor(["fake_command"])
      executor.start()
      mock_popen.assert_called_once_with(
          args=["fake_command"],
          stdout=subprocess.PIPE,
          stdin=subprocess.PIPE,
      )
    mock_process.stdout = io.BytesIO(b'{"placeholder": "output"}\n')
    mock_process.stdin = io.BytesIO()

    response = executor.execute({"meaningless": "filler"})

    self.assertEqual(
        b'{"meaningless": "filler"}\n', mock_process.stdin.getvalue()
    )
    self.assertDictEqual({"placeholder": "output"}, response)

  def test_subprocess_executor_execute_error_output_closed(self):
    mock_process = mock.create_autospec(subprocess.Popen, instance=True)
    with mock.patch.object(
        subprocess, "Popen", autospec=True, return_value=mock_process
    ) as mock_popen:
      executor = server_gunicorn.SubprocessPredictionExecutor(["fake_command"])
      executor.start()

      mock_process.stdout = io.BytesIO()  # empty output simulates closed pipe.
      mock_process.stdin = io.BytesIO()

      with self.assertRaises(RuntimeError) as raised:
        executor.execute({"meaningless": "filler"})
      self.assertEqual(
          raised.exception.args[0], "Executor process output stream closed."
      )
      self.assertEqual(mock_popen.call_count, 2)  # executor restarted.

  def test_subprocess_executor_execute_error_input_broken(self):
    mock_process = mock.create_autospec(subprocess.Popen, instance=True)
    with mock.patch.object(
        subprocess, "Popen", autospec=True, return_value=mock_process
    ) as mock_popen:
      executor = server_gunicorn.SubprocessPredictionExecutor(["fake_command"])
      executor.start()

      mock_process.stdout = io.BytesIO(b'{"placeholder": "output"}\n')
      # Simulate broken pipe.
      mock_process.stdin = mock.create_autospec(io.BytesIO, instance=True)
      mock_process.stdin.write.side_effect = BrokenPipeError

      with self.assertRaises(RuntimeError):
        executor.execute({"meaningless": "filler"})
      self.assertEqual(mock_popen.call_count, 2)  # executor restarted.


if __name__ == "__main__":
  absltest.main()
