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

import http
import os
from unittest import mock

import requests
import requests_mock

from absl.testing import absltest
from serving.serving_framework import server_gunicorn
from serving.serving_framework.triton import server_health_check


class ServerHealthCheckTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["AIP_PREDICT_ROUTE"] = "/fake-predict-route"
    os.environ["AIP_HEALTH_ROUTE"] = "/fake-health-route"

  @requests_mock.Mocker()
  def test_health_route_pass_check(self, mock_requests):
    mock_requests.register_uri(
        "GET",
        "http://localhost:12345/v2/health/ready",
        text="assorted_metadata",
        status_code=http.HTTPStatus.OK,
    )

    executor = mock.create_autospec(
        server_gunicorn.PredictionExecutor,
        instance=True,
    )

    app = server_gunicorn.PredictionApplication(
        executor,
        health_check=server_health_check.TritonServerHealthCheck(
            12345
        ),
    ).load()
    service = app.test_client()

    response = service.get("/fake-health-route")

    self.assertEqual(response.status_code, http.HTTPStatus.OK)
    self.assertEqual(response.text, "ok")

  @requests_mock.Mocker()
  def test_health_route_fail_check(self, mock_requests):
    mock_requests.register_uri(
        "GET",
        "http://localhost:12345/v2/health/ready",
        exc=requests.exceptions.ConnectionError,
    )
    executor = mock.create_autospec(
        server_gunicorn.PredictionExecutor,
        instance=True,
    )

    app = server_gunicorn.PredictionApplication(
        executor,
        health_check=server_health_check.TritonServerHealthCheck(
            12345
        ),
    ).load()
    service = app.test_client()

    response = service.get("/fake-health-route")

    self.assertEqual(response.status_code, http.HTTPStatus.SERVICE_UNAVAILABLE)
    self.assertEqual(response.text, "not ok")


if __name__ == "__main__":
  absltest.main()
